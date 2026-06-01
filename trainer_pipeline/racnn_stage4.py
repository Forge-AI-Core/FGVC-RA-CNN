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


#############################################################################################
# stage4 에서는 classifier1, 2(on scale1, 2), apn1을 고정시킨 채 apn2만 학습시킵니다.
#############################################################################################


################################
# 앞서 학습시킨 모듈들을 불러옵니다.
################################
def prepare_modules(
    classifier1_path: Path,
    classifier2_path: Path,
    apn1_path: Path,
    num_classes: int,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """stage2, stage3의 가중치를 기반으로 classifier2과 classifier3, apn1의 가중치를 불러오는 코드"""
    # 모델 파일 로드 (VRAM 낭비 방지를 위해 CPU로 먼저 로드합니다)
    state_dict1 = torch.load(classifier1_path, map_location='cpu', weights_only=True)
    state_dict2 = torch.load(classifier2_path, map_location='cpu', weights_only=True)
    state_dict_apn1 = torch.load(apn1_path, map_location='cpu', weights_only=True)

    # 분류기 1
    classifier1 = VanillaCNN(num_classes=num_classes)
    classifier1.load_state_dict(state_dict=state_dict1)

    # 분류기 2
    classifier2 = VanillaCNN(num_classes=num_classes)
    # 모델객체 타입은 stage2의 Stage2CropNet 클래스입니다. (classifier와 apn을 포함합니다)
    # 따라서 스페시픽한 classfier를 먼저 추출해주고 추출한 classifier2의 가중치에서 "classifier2." 접두어를 제거합니다.
    c2_weights = {k.replace('classifier2.', ''): v for k, v in state_dict2.items() if k.startswith('classifier2.')}
    classifier2.load_state_dict(state_dict=c2_weights)

    # 분류기 3
    classifier3 = VanillaCNN(num_classes=num_classes)
    # classifier3도 classifier2의 가중치로 똑같이 초기화합니다.
    classifier3.load_state_dict(state_dict=c2_weights)

    # APN 1
    apn1 = APN(in_features=512*10*10)
    apn1_weights = {k.replace('apn.', ''): v for k, v in state_dict_apn1.items() if k.startswith('apn.')}
    apn1.load_state_dict(apn1_weights)

    # APN 2
    apn2 = APN(in_features=512*10*10)

    return classifier1, classifier2, classifier3, apn1, apn2


###################################################################################
# model_architectures 폴더에 정의된 모듈 아키텍쳐들을 Stage4에 필요한 방식으로 재구성합니다.
###################################################################################
class Stage4Model(nn.Module):

    def __init__(self, classifier1: nn.Module, classifier2: nn.Module, classifier3: nn.Module, apn1: nn.Module, apn2: nn.Module):
        super().__init__()

        self.classifier1 = classifier1
        self.feature_extractor1 = classifier1.features
        self.apn1 = apn1

        self.classifier2 = classifier2
        self.feature_extractor2 = classifier2.features
        self.apn2 = apn2

        self.classifier3 = classifier3
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat1 = self.feature_extractor1(x)
        
        tx1, ty1, tl1 = self.apn1(feat1)
        cropped_image1 = crop_image(image=x, tx=tx1, ty=ty1, tl=tl1)

        # 여기서부터 이번 스테이지에 사용될 피쳐
        feat2 = self.feature_extractor2(cropped_image1)
        logits2 = self.classifier2.classifier(feat2)
        
        tx2, ty2, tl2 = self.apn2(feat2)
        cropped_image2 = crop_image(image=cropped_image1, tx=tx2, ty=ty2, tl=tl2)

        logits3 = self.classifier3(cropped_image2)

        return logits2, logits3, cropped_image2


####################
# Stage4의 학습 코드
####################
def train_stage4(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, device: str) -> None:
    """stage4의 학습 코드"""
    # 학습하지 않을 파라미터들 프리징
    for param in model.classifier1.parameters():
        param.requires_grad = False
    for param in model.apn1.parameters():
        param.requires_grad = False
    for param in model.classifier2.parameters():
        param.requires_grad = False
    for param in model.classifier3.parameters():
        param.requires_grad = False

    # 학습모드 결정(.eval은 파라미터와 별개로 drop out, bn 등을 끕니다.)
    model.train()
    model.classifier1.eval()
    model.apn1.eval()
    model.classifier2.eval()
    model.classifier3.eval()

    running_crop_loss = 0.0
    # tqdm pbar
    progress_bar = tqdm(iterable=train_loader, desc="Training")
    for index, batch in enumerate(progress_bar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()

        # mixed precision: bf16이 fp16보다 빠릅니다.
        # 피드포워드와 손실함수 계산까지만 포함
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits2, logits3, _ = model(images)
            crop_loss2 = rank_loss(logits2, logits3, labels=labels)

        # 역전파와 업데이트는 fp32로 진행
        crop_loss2.backward()
        optimizer.step()

        # for loss
        running_crop_loss += crop_loss2.item()
        avg_crop_loss2 = running_crop_loss / (index + 1)

        # set.postfix: 프린트를 일일이 출력하는 대신, 진행바에 표시하여 화면이 길게 늘어지는 것을 방지
        # 프로그레스 바 뒤에 고정되는 것은 인자로 입력한 딕셔너리
        progress_bar.set_postfix({
            "crop_loss2": f"{avg_crop_loss2:.4f}"
        })


#####################
# Stage4의 검증 코드
#####################
def val_stage4(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str) -> float:
    """stage4의 검증 코드"""
    model.eval()

    running_loss = 0.0
    correct2 = 0
    correct3 = 0
    total = 0
    # with torch.no_grad(): 가장 외곽에 위치
    with torch.no_grad():
        progress_bar = tqdm(iterable=val_loader, desc="VAL")
        for index, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits2, logits3, _ = model(images)
                loss2 = criterion(logits2, labels)
                loss3 = criterion(logits3, labels)
                        
            # for loss
            classification_loss = loss2 + loss3            
            running_loss += classification_loss.item()
            avg_loss = running_loss / (index + 1)

            # for accuracy
            total += labels.size(dim=0)
            _, predictions2 = logits2.max(dim=1)
            _, predictions3 = logits3.max(dim=1)
            correct2 += predictions2.eq(labels).sum().item()
            correct3 += predictions3.eq(labels).sum().item()
            acc2 = correct2 / total * 100.0
            acc3 = correct3 / total * 100.0

            # set.postfix: 프린트를 일일이 출력하는 대신, 진행바에 표시하여 화면이 길게 늘어지는 것을 방지
            # 프로그레스 바 뒤에 고정되는 것은 인자로 입력한 딕셔너리
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc2": f"{acc2:.2f}%",
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
        

#######################
# 학습된 APN2를 얻는 함수
#######################
def get_apn2(dataset_name: str, hyperparameter_path: Path) -> None:
    """APN2의 main 코드"""
    # 하이퍼 파라미터 파싱
    hyperparameters = load_hyper_parameters(hyperparameter_path)

    batch_size = int(hyperparameters["training"]["BATCH_SIZE"])
    num_epochs = int(hyperparameters["training"]["NUM_EPOCHS"])
    learning_rate = float(hyperparameters["training"]["LEARNING_RATE"])
    weight_decay = float(hyperparameters["training"]["WEIGHT_DECAY"])
    
    # 디바이스
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터
    train_loader, val_loader, _ = get_dataloaders(dataset_name=dataset_name, batch_size=batch_size)
    num_classes = get_num_classes(dataset_name=dataset_name)
    
    # 모듈
    classifier1, classifier2, classifier3, apn1, apn2 = prepare_modules(
        classifier1_path=f"models/{dataset_name}/{dataset_name}_best_on_stage1.pth",
        classifier2_path=f"models/{dataset_name}/{dataset_name}_best_on_stage3.pth",
        apn1_path=f"models/{dataset_name}/{dataset_name}_best_on_stage2.pth",
        num_classes=num_classes
    )

    # 모델
    model = Stage4Model(
        classifier1=classifier1, 
        classifier2=classifier2, 
        classifier3=classifier3, 
        apn1=apn1, 
        apn2=apn2,
    ).to(device)
    
    # 옵티마이저, 손실함수, 학습속도 스케줄러
    optimizer = optim.AdamW(params=model.apn2.parameters(), lr=learning_rate, weight_decay=weight_decay)
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

        # 학습과 검증
        train_stage4(model, train_loader, optimizer, device)
        acc3 = val_stage4(model, val_loader, criterion, device)

        # 스케줄러의 스텝은 모델 내부가 아닌 루프 내부에서 수행
        scheduler.step()
        if acc3 > best_acc:
            best_acc = acc3
            patience_counter = 0
            Path(f"models/{dataset_name}").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"models/{dataset_name}/{dataset_name}_best_on_stage4.pth")

            print(f"best_acc: {best_acc:.2f}%")
        else:
            patience_counter += 1

            print(f"EarlyStopping patience counter: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:

                print("Early Stopping triggered.")
                print(f"final_acc: {best_acc:.2f}%")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options: CUB-200-2011, FGVC-Aircraft, Stanford-Cars, Iron-Scraps")
    parser.add_argument("--dataset", type=str, required=True, help="데이터셋 이름을 입력하세요.")
    parser.add_argument("--hyperparameter", type=str, default="hyper-parameters.yaml", help="YAML파일 경로")
    args = parser.parse_args()

    get_apn2(
        dataset_name=args.dataset,
        hyperparameter_path=Path(args.hyperparameter),
    )