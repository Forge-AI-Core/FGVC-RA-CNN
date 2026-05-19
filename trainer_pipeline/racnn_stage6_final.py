import yaml
import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from trainer_pipeline.data_loaders import get_dataloaders, get_num_classes
from trainer_pipeline.model_base_architectures.vanilla_cnn import VanillaCNN
from trainer_pipeline.model_base_architectures.apn import APN, crop_image, rank_loss


#######################################################################################
# stage6_final 에서는 모든 모듈의 가중치를 녹인 후 러닝레이트를 줄여서 최종적으로 allign 합니다.
#######################################################################################


################################
# 앞서 학습시킨 모듈들을 불러옵니다.
################################
def prepare_modules(
    classifier1_path: Path, 
    classifier2_path: Path,
    classifier3_path: Path,
    apn1_path: Path,
    apn2_path: Path,
    num_classes: int,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """이전 모든 stage들의 가중치를 기반으로 최종 통합 학습을 위한 모듈들을 로드합니다."""
    # 모델 파일 로드 (VRAM 낭비 방지를 위해 CPU로 먼저 로드합니다)
    cls1_state_dict = torch.load(classifier1_path, map_location='cpu', weights_only=True)
    cls2_state_dict = torch.load(classifier2_path, map_location='cpu', weights_only=True)
    cls3_state_dict = torch.load(classifier3_path, map_location='cpu', weights_only=True)
    apn1_state_dict = torch.load(apn1_path, map_location='cpu', weights_only=True)
    apn2_state_dict = torch.load(apn2_path, map_location='cpu', weights_only=True)

    # 분류기 1
    classifier1 = VanillaCNN(num_classes=num_classes)
    classifier1.load_state_dict(state_dict=cls1_state_dict)

    # 분류기 2
    classifier2 = VanillaCNN(num_classes=num_classes)
    # 모델객체는 stage2의 Stage2Model 클래스(classifier와 apn을 포함)입니다.
    # 따라서 스페시픽한 classfier를 먼저 추출해주고 추출한 classifier2의 가중치에서 "classifier2." 접두어를 제거합니다.
    c2_weights = {k.replace('classifier2.', ''): v for k, v in cls2_state_dict.items() if k.startswith('classifier2.')}
    classifier2.load_state_dict(state_dict=c2_weights)

    # 분류기 3
    classifier3 = VanillaCNN(num_classes=num_classes)
    # 가중치 딕셔너리에서 "classifier3." 접두어를 제거하여 로드합니다.
    c3_weights = {k.replace('classifier3.', ''): v for k, v in cls3_state_dict.items() if k.startswith('classifier3.')}
    classifier3.load_state_dict(state_dict=c3_weights)
    
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


class Stage6Model(nn.Module):

    def __init__(
        self, 
        classifier1: nn.Module, 
        classifier2: nn.Module, 
        classifier3: nn.Module, 
        apn1: nn.Module, 
        apn2: nn.Module,
    ) -> None:
        super().__init__()
        self.classifier1 = classifier1
        self.classifier2 = classifier2
        self.classifier3 = classifier3
        self.apn1 = apn1
        self.apn2 = apn2
        self.feature_extractor1 = self.classifier1.features
        self.feature_extractor2 = self.classifier2.features
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat1 = self.feature_extractor1(x)
        logits1 = self.classifier1.classifier(feat1)
        
        tx1, ty1, tl1 = self.apn1(feat1)
        cropped_image1 = crop_image(image=x, tx=tx1, ty=ty1, tl=tl1)

        feat2 = self.feature_extractor2(cropped_image1)
        logits2 = self.classifier2.classifier(feat2)
        
        tx2, ty2, tl2 = self.apn2(feat2)
        cropped_image2 = crop_image(image=cropped_image1, tx=tx2, ty=ty2, tl=tl2)

        logits3 = self.classifier3(cropped_image2)

        return logits1, logits2, logits3, cropped_image1, cropped_image2


####################
# Stage6의 학습 코드
####################
def train_stage6(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str) -> float:
    """stage6의 학습 코드 (모든 파라미터 업데이트)"""    
    model.train()    

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
            logits1, logits2, logits3, _, _ = model(images)
            classification_loss = criterion(logits1, labels) + criterion(logits2, labels) + criterion(logits3, labels)
            crop_loss1 = rank_loss(logits1, logits2, labels=labels)
            crop_loss2 = rank_loss(logits2, logits3, labels=labels)
            
        # 나머지는 fp32로 진행
        total_loss = classification_loss + crop_loss1 + crop_loss2
        total_loss.backward()
        optimizer.step()

        # for loss
        running_loss += total_loss.item()
        avg_loss = running_loss / (index + 1)

        # set.postfix: 프린트를 일일이 출력하는 대신, 진행바에 표시하여 화면이 길게 늘어지는 것을 방지
        # 프로그레스 바 뒤에 고정되는 것은 인자로 입력한 딕셔너리
        progress_bar.set_postfix({
            "avg_loss": f"{avg_loss:.4f}"
        })

    return avg_loss


#####################
# Stage6의 테스트 코드
#####################
def test_stage6(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float, float, float]:
    """stage6의 테스트 코드 (상세 지표 및 손실 반환)"""
    model.eval()

    running_loss = 0.0
    correct_ensemble = 0
    total = 0
    all_preds = []
    all_labels = []

    # with torch.no_grad(): 가장 외곽에 위치
    with torch.no_grad():
        progress_bar = tqdm(iterable=test_loader, desc="TEST")
        for index, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # mixed precision: 개별 손실 계산까지 포함
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits1, logits2, logits3, _, _ = model(images)
                classification_loss = criterion(logits1, labels) + criterion(logits2, labels) + criterion(logits3, labels)
                crop_loss1 = rank_loss(logits1, logits2, labels=labels)
                crop_loss2 = rank_loss(logits2, logits3, labels=labels)
            
            # FP32로 단순 합산만 수행 (수치 안정성 확보)
            total_loss = classification_loss + crop_loss1 + crop_loss2
            running_loss += total_loss.item()
            avg_loss = running_loss / (index + 1)
            total += labels.size(dim=0)
            
            # 단순 평균 앙상블
            ensemble_logits = (logits1 + logits2 + logits3) / 3.0
            _, predictions_ensemble = ensemble_logits.max(dim=1)
            correct_ensemble += predictions_ensemble.eq(labels).sum().item()

            all_preds.extend(predictions_ensemble.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            acc_ens = correct_ensemble / total * 100.0

            # 진행바 업데이트
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc_ens": f"{acc_ens:.2f}%",        
            })

    # 정밀 지표 계산 (Macro 평균)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc_ens, precision, recall, f1


###################################
# 메인 코드에서 호출할 시각화 함수 정의
###################################
def visualize_results(history: dict[str, list[float]], dataset_name: str) -> None:
    """학습 결과(Loss, Acc, F1, P&R)를 시각화하여 저장합니다."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"RA-CNN Stage 6 Training Results ({dataset_name})", fontsize=16)

    # 1. Loss (Train & Test)
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, history["test_loss"], label="Test Loss", marker='x')
    plt.title("Training & Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 2. Accuracy (Ensemble)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["test_acc"], color='green', label="Ensemble Acc", marker='s')
    plt.title("Ensemble Accuracy (Test)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # 3. F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["test_f1"], color='red', label="F1 Score", marker='d')
    plt.title("F1 Score (Test)")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    # 4. Precision & Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["test_precision"], label="Precision", marker='^')
    plt.plot(epochs, history["test_recall"], label="Recall", marker='v')
    plt.title("Precision & Recall (Test)")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 결과 저장 폴더 생성 및 저장
    Path(f"results/{dataset_name}").mkdir(parents=True, exist_ok=True)
    save_path = f"results/{dataset_name}/stage6_{dataset_name}_learning_curves.png"
    plt.savefig(save_path)

    print(f"\nLearning curves saved to: {save_path}")

    plt.close()


####################
# 하이퍼 파라미터 로드
####################
def load_hyper_parameters(hyperparameter_path: Path) -> dict[str, Any]:
    """하이퍼 파라미터를 불러오는 코드"""
    with open(hyperparameter_path, "r", encoding="utf-8") as file:
        hyperparameters = yaml.safe_load(file)

    return hyperparameters
        

##########################
# 최종 통합 모델을 얻는 함수
##########################
def get_racnn(dataset_name: str, hyperparameter_path: Path) -> None:
    """racnn_stage6_final의 main 코드"""
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
        classifier3_path=f"models/{dataset_name}/{dataset_name}_best_on_stage5.pth",
        num_classes=num_classes
    )

    # 모델
    model = Stage6Model(classifier1=classifier1, classifier2=classifier2, classifier3=classifier3, apn1=apn1, apn2=apn2).to(device)
    
    # 옵티마이저, 손실함수, 학습속도 스케줄러
    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

    # 기록용 딕셔너리
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": []
    }

    # 루프 정의
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = hyperparameters["training"]["EARLY_STOP_PATIENCE"]
    
    for epoch in range(num_epochs):
        current_lr = scheduler.get_last_lr()[0]

        print(f"Current Learning Rate: {current_lr:.2e}")
        print(f"Epoch: {epoch+1}/{num_epochs}")

        # 학습과 테스트
        train_loss = train_stage6(model, train_loader, criterion, optimizer, device)
        test_loss, acc_ens, precision, recall, f1 = test_stage6(model, test_loader, criterion, device)

        # 기록 저장
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(acc_ens)
        history["test_precision"].append(precision)
        history["test_recall"].append(recall)
        history["test_f1"].append(f1)

        # 스케줄러의 스텝은 모델 내부가 아닌 루프 내부에서 수행
        scheduler.step()
        
        if acc_ens > best_acc:
            best_acc = acc_ens
            patience_counter = 0
            Path(f"models/{dataset_name}").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"models/{dataset_name}/{dataset_name}_best_on_stage6.pth")

            print(f"best_acc: {best_acc:.2f}% (P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f})")
        else:
            patience_counter += 1

            print(f"EarlyStopping patience counter: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:

                print("Early Stopping triggered.")
                print(f"final_acc: {best_acc:.2f}%")
                break

    # 학습 완료 후 시각화 함수 호출
    visualize_results(history, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options: CUB-200-2011, FGVC-Aircraft, Stanford-Cars")
    parser.add_argument("--dataset", type=str, required=True, help="데이터셋 이름을 입력하세요.")
    parser.add_argument("--hyperparameter", type=str, default="hyper-parameters.yaml", help="YAML파일 경로")
    args = parser.parse_args()

    get_racnn(
        dataset_name=args.dataset,
        hyperparameter_path=Path(args.hyperparameter),
    )