import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from trainer_pipeline.data_loaders import get_dataloaders, get_num_classes, get_test_dataloader
from trainer_pipeline.model_base_architectures.apn import rank_loss
from trainer_pipeline.model_base_architectures.ra_cnn import RACNN, prepare_modules
from trainer_pipeline.metrics import visualize_results, evaluate_and_save_metrics, evaluate_with_custom_threshold


#######################################################################################
# stage6_final 에서는 모든 모듈의 가중치를 녹인 후 러닝레이트를 줄여서 최종적으로 allign 합니다.
#######################################################################################


################################
# 앞서 학습시킨 모듈들을 불러옵니다.
# (prepare_modules 함수는 trainer_pipeline.model_base_architectures.ra_cnn 으로 이동되었으며,
# 이전 모든 stage들의 가중치를 기반으로 최종 통합 학습을 위한 모듈들을 로드합니다.)
# (VRAM 낭비 방지를 위해 CPU로 먼저 로드합니다)
# (분류기 1, 분류기 2, 분류기 3, APN 1, APN 2 로드 로직 포함)
################################


# RACNN 클래스는 trainer_pipeline.model_base_architectures.ra_cnn 으로 이동되었습니다.
# 최종 3-scale 통합 RA-CNN 모델 구조를 정의합니다.


####################
# Stage6의 학습 코드
####################
def train_stage6(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
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
            classification_loss = (
                criterion(logits1, labels)
                + criterion(logits2, labels)
                + criterion(logits3, labels)
            )
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
        progress_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

    return avg_loss


#####################
# Stage6의 검증 코드
#####################
def val_stage6(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float, float, float]:
    """stage6의 검증 코드 (상세 지표 및 손실 반환)"""
    model.eval()

    running_loss = 0.0
    correct_ensemble = 0
    total = 0
    all_preds = []
    all_labels = []

    # with torch.no_grad(): 가장 외곽에 위치
    with torch.no_grad():
        progress_bar = tqdm(iterable=val_loader, desc="VAL")
        for index, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # mixed precision: 개별 손실 계산까지 포함
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits1, logits2, logits3, _, _ = model(images)
                classification_loss = (
                    criterion(logits1, labels)
                    + criterion(logits2, labels)
                    + criterion(logits3, labels)
                )
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
            progress_bar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "acc_ens": f"{acc_ens:.2f}%",
                }
            )

    # 정밀 지표 계산 (Macro 평균)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc_ens, precision, recall, f1


###################################
# 메인 코드에서 호출할 시각화 함수 정의
# (visualize_results 함수는 trainer_pipeline.metrics로 이동되었으며,
# 학습 결과(Loss, Acc, F1, P&R)를 시각화하여 저장합니다.)
# 1. Loss (Train & Val), 2. Accuracy (Ensemble), 3. F1 Score, 4. Precision & Recall 시각화 로직
# 결과 저장 폴더 생성 및 저장 로직 포함
###################################


# evaluate_and_save_metrics 함수는 trainer_pipeline.metrics로 이동되었습니다.
# 최종 베스트 모델에 대해 클래스별 지표 및 혼동 행렬을 계산하여 출력하고 저장합니다.
# 클래스명 추출, 1. classification report (클래스별 프리시전, 리콜, 종합 어큐러시 포함) 저장,
# 2. Confusion Matrix 계산 및 시각화 저장 로직 포함


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
    train_loader, val_loader, _ = get_dataloaders(
        dataset_name=dataset_name, batch_size=batch_size
    )
    num_classes = get_num_classes(dataset_name=dataset_name)

    # 모듈
    classifier1, classifier2, classifier3, apn1, apn2 = prepare_modules(
        classifier1_path=f"models/{dataset_name}/{dataset_name}_best_on_stage1.pth",
        apn1_path=f"models/{dataset_name}/{dataset_name}_best_on_stage2.pth",
        classifier2_path=f"models/{dataset_name}/{dataset_name}_best_on_stage3.pth",
        apn2_path=f"models/{dataset_name}/{dataset_name}_best_on_stage4.pth",
        classifier3_path=f"models/{dataset_name}/{dataset_name}_best_on_stage5.pth",
        num_classes=num_classes,
    )

    # 모델
    model = RACNN(
        classifier1=classifier1,
        classifier2=classifier2,
        classifier3=classifier3,
        apn1=apn1,
        apn2=apn2,
    ).to(device)

    # 옵티마이저, 손실함수, 학습속도 스케줄러
    optimizer = optim.AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs
    )

    # 기록용 딕셔너리
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    # 루프 정의
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = hyperparameters["training"]["EARLY_STOP_PATIENCE"]

    for epoch in range(num_epochs):
        current_lr = scheduler.get_last_lr()[0]

        print(f"Current Learning Rate: {current_lr:.2e}")
        print(f"Epoch: {epoch+1}/{num_epochs}")

        # 학습과 검증
        train_loss = train_stage6(model, train_loader, criterion, optimizer, device)
        val_loss, acc_ens, precision, recall, f1 = val_stage6(
            model, val_loader, criterion, device
        )

        # 기록 저장
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc_ens)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)

        # 스케줄러의 스텝은 모델 내부가 아닌 루프 내부에서 수행
        scheduler.step()

        if acc_ens > best_acc:
            best_acc = acc_ens
            patience_counter = 0
            Path(f"models/{dataset_name}").mkdir(parents=True, exist_ok=True)
            torch.save(
                model.state_dict(),
                f"models/{dataset_name}/{dataset_name}_best_on_stage6.pth",
            )

            print(
                f"best_acc: {best_acc:.2f}% (P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f})"
            )
        else:
            payout_counter = patience_counter + 1
            patience_counter = payout_counter

            print(
                f"EarlyStopping patience counter: {patience_counter}/{early_stop_patience}"
            )
            if patience_counter >= early_stop_patience:
                print("Early Stopping triggered.")
                print(f"final_acc: {best_acc:.2f}%")
                break

    # 학습 완료 후 시각화 함수 호출
    visualize_results(history, dataset_name)

    # 베스트 가중치를 로드하여 최종 검증 보고서 및 혼동 행렬 생성
    best_model_path = Path(f"models/{dataset_name}/{dataset_name}_best_on_stage6.pth")
    if best_model_path.exists():
        print(
            "\n[평가] 베스트 모델의 가중치를 불러와 최종 평가(클래스별 지표 및 혼동 행렬)를 시작합니다..."
        )
        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )
        evaluate_and_save_metrics(
            model=model,
            val_loader=val_loader,
            device=device,
            dataset_name=dataset_name,
            num_classes=num_classes,
            prefix="val",
        )
        
        # 새로운 threshold 적용하여 test set을 평가
        print("\n[평가] Custom Threshold를 적용한 Test 셋 평가를 진행합니다...")
        if dataset_name == "Iron-Scraps":
            test_loader = get_test_dataloader(dataset_name=dataset_name, batch_size=batch_size)
            evaluate_with_custom_threshold(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                dataset_name=dataset_name,
                danger_class_idx=1,
                target_recall=0.9
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Options: CUB-200-2011, FGVC-Aircraft, Stanford-Cars, Iron-Scraps"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="데이터셋 이름을 입력하세요."
    )
    parser.add_argument(
        "--hyperparameter",
        type=str,
        default="hyper-parameters.yaml",
        help="YAML파일 경로",
    )
    args = parser.parse_args()

    get_racnn(
        dataset_name=args.dataset,
        hyperparameter_path=Path(args.hyperparameter),
    )
