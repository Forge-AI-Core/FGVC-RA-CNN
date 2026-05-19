from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

import yaml
import argparse

import torch
from torch import nn, optim
from tqdm import tqdm

from trainer_pipeline.data_loaders import get_dataloaders, get_num_classes
from trainer_pipeline.model_base_architectures.vanilla_cnn import VanillaCNN
from trainer_pipeline.model_base_architectures.apn import APN, crop_image, rank_loss


###################################################
# stage5 에서는 나머지 4개 모듈을 모두 고정시킨 채,
# Scale3 이미지로 classifier3 분류기만 학습시킵니다.
###################################################


#####################################
# 앞서 학습시킨 모듈들을 불러옵니다.
#####################################
def prepare_modules(
    classifier1_path: Path, 
    classifier2_path: Path,
    apn1_path: Path,
    apn2_path: Path,
    num_classes: int,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """이전 스테이지들의 가중치를 기반으로 최종 앙상블을 위한 5개의 모듈을 모두 로드합니다."""
    # 모델 파일 로드 (VRAM 낭비 방지를 위해 CPU로 먼저 로드합니다)
    cls1_state_dict = torch.load(classifier1_path, map_location='cpu', weights_only=True)
    cls2_state_dict = torch.load(classifier2_path, map_location='cpu', weights_only=True)
    apn1_state_dict = torch.load(apn1_path, map_location='cpu', weights_only=True)
    apn2_state_dict = torch.load(apn2_path, map_location='cpu', weights_only=True)

    # 분류기 1
    classifier1 = VanillaCNN(num_classes=num_classes)
    classifier1.load_state_dict(state_dict=cls1_state_dict)

    # 분류기 2
    classifier2 = VanillaCNN(num_classes=num_classes)
    # 모델객체 타입은 stage2의 Stage2Model 클래스입니다. (classifier와 apn을 포함합니다)
    # 따라서 스페시픽한 classfier를 먼저 추출해주고 추출한 classifier2의 가중치에서 "classifier2." 접두어를 제거합니다.
    c2_weights = {k.replace('classifier2.', ''): v for k, v in cls2_state_dict.items() if k.startswith('classifier2.')}
    classifier2.load_state_dict(state_dict=c2_weights)

    # 분류기 3
    classifier3 = VanillaCNN(num_classes=num_classes)
    classifier3.load_state_dict(state_dict=c2_weights)

    # APN 1
    apn1 = APN(in_features=512*10*10)
    # 파이토치의 state_dict는 {"레이어 이름": "가중치 텐서"} 형태의 파이썬 딕셔너리입니다.
    # 키와 밸류는 각각 다음을 의미합니다:
    # 키: 레이어의 이름 (예: 'apn.fc1.weight', 'apn.fc2.bias')
    # 밸류: 해당 레이어에 저장된 실제 가중치 숫자들(Tensor)
    apn1_weights = {k.replace('apn.', ''): v for k, v in apn1_state_dict.items() if k.startswith('apn.')}
    apn1.load_state_dict(apn1_weights)

    # APN 2
    apn2 = APN(in_features=512*10*10)
    # 가중치 딕셔너리에서 "apn2." 접두어를 제거하여 로드합니다.
    apn2_weights = {k.replace('apn2.', ''): v for k, v in apn2_state_dict.items() if k.startswith('apn2.')}
    apn2.load_state_dict(apn2_weights)

    return classifier1, classifier2, classifier3, apn1, apn2


###################################################################################
# model_architectures 폴더에 정의된 모듈 아키텍쳐들을 Stage5에 필요한 방식으로 재구성합니다.
###################################################################################
class Stage5Model(nn.Module):

    def __init__(
        self, 
        classifier1: nn.Module, 
        classifier2: nn.Module, 
        classifier3: nn.Module, 
        apn1: nn.Module, 
        apn2: nn.Module,
    ):
        super().__init__()

        self.classifier1 = classifier1
        self.feature_extractor1 = classifier1.features
        self.apn1 = apn1

        self.classifier2 = classifier2
        self.feature_extractor2 = classifier2.features
        self.apn2 = apn2

        self.classifier3 = classifier3
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat1 = self.feature_extractor1(x)
        
        tx1, ty1, tl1 = self.apn1(feat1)
        cropped_image1 = crop_image(image=x, tx=tx1, ty=ty1, tl=tl1)

        feat2 = self.feature_extractor2(cropped_image1)
        
        tx2, ty2, tl2 = self.apn2(feat2)
        cropped_image2 = crop_image(image=cropped_image1, tx=tx2, ty=ty2, tl=tl2)

        logits3 = self.classifier3(cropped_image2)

        return logits3, cropped_image2


####################
# Stage5의 학습 코드
####################
def train_stage5(
    model: nn.Module, 
    train_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: str,
) -> float:
    """stage5의 학습 코드"""
    # 학습하지 않을 파라미터들 프리징
    for param in model.classifier1.parameters():
        param.requires_grad = False
    for param in model.apn1.parameters():
        param.requires_grad = False
    for param in model.classifier2.parameters():
        param.requires_grad = False
    for param in model.apn2.parameters():
        param.requires_grad = False

    # 학습모드 결정(.eval은 파라미터와 별개로 drop out, bn 등을 끕니다.)
    model.train()
    model.classifier1.eval()
    model.apn1.eval()
    model.classifier2.eval()
    model.apn2.eval()

    running_loss = 0.0
    # tqdm pbar
    progress_bar = tqdm(iterable=train_loader, desc="Training")
    for index, batch in enumerate(progress_bar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()

        # mixed precision: bf16이 fp16보다 빠릅니다.
        # 피드포워드와 손실함수 계산까지만 포함
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits3, _ = model(images)
            classification_loss = criterion(logits3, labels)

        # 역전파와 업데이트는 fp32로 진행
        classification_loss.backward()
        optimizer.step()
        
        # for loss
        running_loss += classification_loss.item()
        avg_loss = running_loss / (index + 1)

        # set.postfix: 프린트를 일일이 출력하는 대신, 진행바에 표시하여 화면이 길게 늘어지는 것을 방지
        # 프로그레스 바 뒤에 고정되는 것은 인자로 입력한 딕셔너리
        progress_bar.set_postfix({
            "avg_loss": f"{avg_loss:.4f}"
        })

    return avg_loss
        

#####################
# Stage5의 테스트 코드
#####################
def test_stage5(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """stage5의 테스트 코드"""
    model.eval()

    running_loss = 0.0
    correct3 = 0
    total = 0
    # with torch.no_grad(): 가장 외곽에 위치
    with torch.no_grad():
        progress_bar = tqdm(iterable=test_loader, desc="TEST")
        for index, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits3, _ = model(images)
                classification_loss = criterion(logits3, labels)
                        
            # for loss
            running_loss += classification_loss.item()
            avg_loss = running_loss / (index + 1)

            # for accuracy
            total += labels.size(dim=0)
            _, predictions3 = logits3.max(dim=1)
            correct3 += predictions3.eq(labels).sum().item()
            acc3 = correct3 / total * 100.0

            # set.postfix: 프린트를 일일이 출력하는 대신, 진행바에 표시하여 화면이 길게 늘어지는 것을 방지
            # 프로그레스 바 뒤에 고정되는 것은 인자로 입력한 딕셔너리
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc3": f"{acc3:.2f}%",        
            })

    return acc3


####################
# 하이퍼 파라미터 로드
####################
def load_hyper_parameters(hyperparameter_path: Path) -> dict[str, Any]:
    """하이퍼 파라미터를 불러오는 코드"""
    with open(hyperparameter_path, "r", encoding="utf-8") as file:
        hyperparameters = yaml.safe_load(file)

    return hyperparameters
        

##############################
# 학습된 Classifier3를 얻는 함수
##############################
def get_classifier3(dataset_name: str, hyperparameter_path: Path) -> None:
    """classifier3의 main 코드"""
    # 하이퍼 파라미터 파싱
    hyperparameters = load_hyper_parameters(hyperparameter_path)

    batch_size = int(hyperparameters["training"]["BATCH_SIZE"])
    num_epochs = int(hyperparameters["training"]["NUM_EPOCHS"])
    learning_rate = float(hyperparameters["training"]["LEARNING_RATE"])
    weight_decay = float(hyperparameters["training"]["WEIGHT_DECAY"])
    
    # 디바이스
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터
    train_loader, test_loader, _ = get_dataloaders(dataset_name=dataset_name, batch_size=batch_size)
    num_classes = get_num_classes(dataset_name=dataset_name)
    
    # 모듈
    classifier1, classifier2, classifier3, apn1, apn2 = prepare_modules(
        classifier1_path=f"models/{dataset_name}/{dataset_name}_best_on_stage1.pth", 
        apn1_path=f"models/{dataset_name}/{dataset_name}_best_on_stage2.pth",
        classifier2_path=f"models/{dataset_name}/{dataset_name}_best_on_stage3.pth",
        apn2_path=f"models/{dataset_name}/{dataset_name}_best_on_stage4.pth",
        num_classes=num_classes
    )

    # 모델
    model = Stage5Model(
        classifier1=classifier1, 
        classifier2=classifier2, 
        classifier3=classifier3, 
        apn1=apn1, 
        apn2=apn2,
    ).to(device)
    
    # 옵티마이저, 손실함수, 학습 속도 스케줄러
    optimizer = optim.AdamW(
        params=model.classifier3.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

    # 루프 정의
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = hyperparameters["training"]["EARLY_STOP_PATIENCE"]
    for epoch in range(num_epochs):
        current_lr = scheduler.get_last_lr()[0]

        print(f"Current Learning Rate: {current_lr:.2e}")
        print(f"Epoch: {epoch+1}/{num_epochs}")

        # 학습과 테스트
        train_stage5(model, train_loader, criterion, optimizer, device)
        acc3 = test_stage5(model, test_loader, criterion, device)

        # 스케줄러의 스텝은 모델 내부가 아닌 루프 내부에서 수행
        scheduler.step()
        if acc3 > best_acc:
            best_acc = acc3
            patience_counter = 0
            Path(f"models/{dataset_name}").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"models/{dataset_name}/{dataset_name}_best_on_stage5.pth")

            print(f"best_acc: {best_acc:.2f}%")
        else:
            patience_counter += 1

            print(f"EarlyStopping patience counter: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:

                print("Early Stopping triggered.")
                print(f"final_acc: {best_acc:.2f}%")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options: CUB-200-2011, FGVC-Aircraft, Stanford-Cars")
    parser.add_argument("--dataset", type=str, required=True, help="데이터셋 이름을 입력하세요.")
    parser.add_argument("--hyperparameter", type=str, default="hyper-parameters.yaml", help="YAML파일 경로")
    args = parser.parse_args()

    get_classifier3(
        dataset_name=args.dataset,
        hyperparameter_path=Path(args.hyperparameter),
    )