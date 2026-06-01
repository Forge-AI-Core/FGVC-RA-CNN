from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def visualize_results(history: dict[str, list[float]], dataset_name: str) -> None:
    """학습 결과(Loss, Acc, F1, P&R)를 시각화하여 저장합니다."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 10))
    plt.suptitle(f"RA-CNN Stage 6 Training Results ({dataset_name})", fontsize=16)

    # 1. Loss (Train & Val)
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="x")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 2. Accuracy (Ensemble)
    plt.subplot(2, 2, 2)
    plt.plot(
        epochs, history["val_acc"], color="green", label="Ensemble Acc", marker="s"
    )
    plt.title("Ensemble Accuracy (Val)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # 3. F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["val_f1"], color="red", label="F1 Score", marker="d")
    plt.title("F1 Score (Val)")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    # 4. Precision & Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["val_precision"], label="Precision", marker="^")
    plt.plot(epochs, history["val_recall"], label="Recall", marker="v")
    plt.title("Precision & Recall (Val)")
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


def evaluate_and_save_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    dataset_name: str,
    num_classes: int,
    prefix: str = "val",
) -> tuple[float, float, float, float, float]:
    """최종 베스트 모델에 대해 클래스별 지표 및 혼동 행렬을 계산하여 출력하고 저장합니다."""
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    prefix_upper = prefix.upper()
    prefix_kor = "검증" if prefix == "val" else "테스트"

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Final Evaluation ({prefix_upper})")
        for index, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(images)
                # 모델 아키텍처에 따라 리턴 값의 개수가 다를 수 있으므로 분기 처리합니다.
                if isinstance(outputs, tuple):
                    if len(outputs) == 5:
                        logits1, logits2, logits3, _, _ = outputs
                        ensemble_logits = (logits1 + logits2 + logits3) / 3.0
                    elif len(outputs) == 3:
                        ensemble_logits, _, _ = outputs
                    else:
                        ensemble_logits = outputs[0]
                else:
                    ensemble_logits = outputs

                loss = criterion(ensemble_logits, labels)

            running_loss += loss.item()
            total += labels.size(dim=0)
            _, predictions = ensemble_logits.max(dim=1)
            correct += predictions.eq(labels).sum().item()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            avg_loss = running_loss / (index + 1)
            accuracy = (correct / total) * 100.0
            progress_bar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.2f}%"}
            )

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # 클래스명 추출
    if hasattr(val_loader.dataset, "classes"):
        class_names = val_loader.dataset.classes
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # 1. classification report (클래스별 프리시전, 리콜, 종합 어큐러시 포함)
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"              [최종 {prefix_kor} 셋 평가 보고서]")
    print("=" * 60)
    print(report_str)
    print("=" * 60 + "\n")

    # 결과 폴더 생성
    results_dir = Path(f"results/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 보고서 텍스트 파일 저장
    report_path = results_dir / f"{prefix}_metrics_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f" 데이터셋: {dataset_name} ({prefix_kor})\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_str)
    print(f"📊 상세 메트릭 보고서가 저장되었습니다: {report_path}")

    # 2. Confusion Matrix 계산 및 시각화
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({dataset_name})", fontsize=16)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # 셀 내부에 숫자 텍스트 표시
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    cm_path = results_dir / f"{prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"📸 혼동 행렬 시각화 이미지가 저장되었습니다: {cm_path}\n")

    return avg_loss, accuracy, precision, recall, f1
