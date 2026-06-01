import os
import yaml
import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from trainer_pipeline.data_loaders import get_dataloaders, get_num_classes
from trainer_pipeline.model_base_architectures.vanilla_cnn import VanillaCNN
from trainer_pipeline.model_base_architectures.apn import APN, crop_image, rank_loss


#############################################################################################
# stage2 에서는 classifier1(on scale1), classifier2(on scale2)를 고정시킨 채 apn1만 학습시킵니다.
#############################################################################################


#####################################
# 앞서 학습시킨 분류기 모듈을 불러옵니다.
#####################################
def prepare_classifiers(classifier1_path: Path, num_classes: int) -> tuple[nn.Module, nn.Module]:
    """stage1의 가중치를 기반으로 classifier1과 classifier2의 가중치를 불러오는 코드"""
    state_dict = torch.load(classifier1_path, map_location='cpu', weights_only=True)

    classifier1 = VanillaCNN(num_classes=num_classes)
    classifier1.load_state_dict(state_dict=state_dict)

    classifier2 = VanillaCNN(num_classes=num_classes)
    classifier2.load_state_dict(state_dict=state_dict)

    return classifier1, classifier2


###################################################################################
# model_architectures 폴더에 정의된 모듈 아키텍쳐들을 Stage2에 필요한 방식으로 재구성합니다.
###################################################################################
class Stage2Model(nn.Module):

    def __init__(self, classifier1: nn.Module, classifier2: nn.Module):
        super().__init__()

        self.classifier1 = classifier1
        self.feature_extractor1 = classifier1.features
        self.apn = APN(in_features=512*7*7)
        self.classifier2 = classifier2
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat1 = self.feature_extractor1(x)
        logits1 = self.classifier1.classifier(feat1)
        
        tx, ty, tl = self.apn(feat1)
        cropped_image = crop_image(image=x, tx=tx, ty=ty, tl=tl)

        logits2 = self.classifier2(cropped_image)

        return logits1, logits2, cropped_image


####################
# Stage2의 학습 코드
####################
def train_stage2(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, device: str) -> None:
    """stage2의 학습 코드"""
    # 학습하지 않을 파라미터들 프리징
    for param in model.classifier1.parameters():
        param.requires_grad = False
    for param in model.classifier2.parameters():
        param.requires_grad = False

    # 학습모드 결정(.eval은 파라미터와 별개로 drop out, bn 등을 끕니다.)
    model.train()
    model.classifier1.eval()
    model.classifier2.eval()

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
            logits1, logits2, _ = model(images)
            crop_loss = rank_loss(logits1, logits2, labels=labels)

        # 역전파와 업데이트는 fp32로 진행
        crop_loss.backward()
        optimizer.step()
        
        # for loss
        running_crop_loss += crop_loss.item()
        avg_crop_loss = running_crop_loss / (index + 1)

        # set.postfix: 프린트를 일일이 출력하는 대신, 진행바에 표시하여 화면이 길게 늘어지는 것을 방지
        # 프로그레스 바 뒤에 고정되는 것은 인자로 입력한 딕셔너리
        progress_bar.set_postfix({
            "crop_loss": f"{avg_crop_loss:.4f}"
        })
        

#####################
# Stage2의 검증 코드
#####################
def val_stage2(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str) -> float:
    """stage2의 검증 코드"""
    model.eval()

    running_loss = 0.0
    correct1 = 0
    correct2 = 0
    total = 0
    # with torch.no_grad(): 가장 외곽에 위치
    with torch.no_grad():
        progress_bar = tqdm(iterable=val_loader, desc="VAL")
        for index, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits1, logits2, _ = model(images)
                loss1 = criterion(logits1, labels)
                loss2 = criterion(logits2, labels)
                        
            # for loss
            classification_loss = loss1 + loss2            
            running_loss += classification_loss.item()
            avg_loss = running_loss / (index + 1)

            # for accuracy
            total += labels.size(dim=0)
            _, predictions1 = logits1.max(dim=1)
            _, predictions2 = logits2.max(dim=1)
            correct1 += predictions1.eq(labels).sum().item()
            correct2 += predictions2.eq(labels).sum().item()
            acc1 = correct1 / total * 100.0
            acc2 = correct2 / total * 100.0

            # set.postfix: 프린트를 일일이 출력하는 대신, 진행바에 표시하여 화면이 길게 늘어지는 것을 방지
            # 프로그레스 바 뒤에 고정되는 것은 인자로 입력한 딕셔너리
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc1": f"{acc1:.2f}%",
                "acc2": f"{acc2:.2f}%",
            })

    return acc2


####################
# 하이퍼 파라미터 로드
####################
def load_hyper_parameters(hyperparameter_path: Path) -> dict[str, Any]:
    """하이퍼 파라미터를 불러오는 코드"""
    with open(hyperparameter_path, "r", encoding="utf-8") as file:
        hyperparameters = yaml.safe_load(file)

    return hyperparameters
        


#######################
# 학습된 APN을 얻는 함수
#######################
def get_apn1(dataset_name: str, hyperparameter_path: Path) -> None:
    """APN1의 main 코드"""
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
    classifier1, classifier2 = prepare_classifiers(classifier1_path=f"models/{dataset_name}/{dataset_name}_best_on_stage1.pth", num_classes=num_classes)

    # 모델
    model = Stage2Model(classifier1=classifier1, classifier2=classifier2).to(device)
    
    # 옵티마이저, 손실함수, 학습속도 스케줄러
    optimizer = optim.AdamW(params=model.apn.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        train_stage2(model, train_loader, optimizer, device)
        acc2 = val_stage2(model, val_loader, criterion, device)

        # 스케줄러의 스텝은 모델 내부가 아닌 루프 내부에서 수행
        scheduler.step()
        
        if acc2 > best_acc:
            best_acc = acc2
            patience_counter = 0
            Path(f"models/{dataset_name}").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"models/{dataset_name}/{dataset_name}_best_on_stage2.pth")

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

    get_apn1(
        dataset_name=args.dataset,
        hyperparameter_path=Path(args.hyperparameter),
    )