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
    precision_recall_curve,
    auc,
    matthews_corrcoef
)
import torch.nn.functional as F


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


def evaluate_with_custom_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    dataset_name: str,
    danger_class_idx: int = 1,
    target_recall: float = 0.9,
) -> None:
    """Val 셋을 기준으로 특정 클래스의 Recall이 목표치에 도달하는 Threshold를 찾고, 이를 Test 셋에 적용합니다."""
    model.eval()
    
    def get_predictions_and_labels(loader: DataLoader, desc: str):
        all_probs = []
        all_labels = []
        with torch.no_grad():
            progress_bar = tqdm(loader, desc=desc)
            for batch in progress_bar:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    outputs = model(images)
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
                        
                probs = F.softmax(ensemble_logits.float(), dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                
        return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)

    print("\n[Custom Evaluation] Validation 데이터셋에 대해 확률을 계산합니다...")
    val_probs, val_labels = get_predictions_and_labels(val_loader, "Val Probs")
    
    danger_probs_val = val_probs[:, danger_class_idx].numpy()
    val_labels_np = val_labels.numpy()
    y_true_danger_val = (val_labels_np == danger_class_idx).astype(int)
    
    # Calculate PR curve for danger class
    precisions, recalls, thresholds = precision_recall_curve(y_true_danger_val, danger_probs_val)
    
    # Find threshold where recall >= target_recall (closest to target_recall)
    valid_idx = np.where(recalls >= target_recall)[0]
    if len(valid_idx) == 0:
        best_idx = np.argmin(np.abs(recalls - target_recall))
    else:
        # valid_idx 중 recall이 target_recall에 가장 가까운(가장 작은 차이) 인덱스 선택
        # recalls는 감소하는 형태 (precisions, recalls, thresholds 배열 특성상)
        best_idx = valid_idx[-1]
        
    threshold = thresholds[min(best_idx, len(thresholds)-1)]
    print(f"\n[Custom Evaluation] 발견된 Danger Threshold (Target Recall={target_recall}): {threshold:.4f}")
    
    # Function to apply threshold and calculate metrics
    def calculate_metrics_with_threshold(probs, labels_np, thresh, split_name):
        danger_probs = probs[:, danger_class_idx].numpy()
        y_true_danger = (labels_np == danger_class_idx).astype(int)
        
        # Determine predictions based on threshold
        preds = np.argmax(probs.numpy(), axis=1)
        # Apply custom threshold for danger class
        danger_mask = danger_probs >= thresh
        preds[danger_mask] = danger_class_idx
        
        for i in range(len(preds)):
            if not danger_mask[i] and preds[i] == danger_class_idx:
                probs_copy = probs[i].numpy().copy()
                probs_copy[danger_class_idx] = -1.0
                preds[i] = np.argmax(probs_copy)
                
        # Calculate metrics
        acc = np.mean(preds == labels_np)
        
        y_pred_danger = (preds == danger_class_idx).astype(int)
        danger_precision = precision_score(y_true_danger, y_pred_danger, zero_division=0)
        danger_recall = recall_score(y_true_danger, y_pred_danger, zero_division=0)
        mcc = matthews_corrcoef(y_true_danger, y_pred_danger)
        
        # PR AUC for danger class
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_danger, danger_probs)
        danger_pr_auc = auc(recall_curve, precision_curve)
        
        report = (
            f"--- {split_name} Set Metrics (Threshold: {thresh:.4f}) ---\n"
            f"Accuracy:        {acc:.4f}\n"
            f"Danger Recall:   {danger_recall:.4f}\n"
            f"Danger Precision:{danger_precision:.4f}\n"
            f"Danger MCC:      {mcc:.4f}\n"
            f"Danger PR AUC:   {danger_pr_auc:.4f}\n"
        )
        print(report)
        return report

    val_report = calculate_metrics_with_threshold(val_probs, val_labels_np, threshold, "Validation")
    
    print("\n[Custom Evaluation] Test 데이터셋에 대해 확률을 계산합니다...")
    test_probs, test_labels = get_predictions_and_labels(test_loader, "Test Probs")
    test_report = calculate_metrics_with_threshold(test_probs, test_labels.numpy(), threshold, "Test")
    
    # Save the reports
    results_dir = Path(f"results/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "test_custom_metrics_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f" 데이터셋: {dataset_name} (Custom Threshold Evaluation)\n")
        f.write(f" Target Danger Recall: {target_recall}\n")
        f.write("=" * 60 + "\n\n")
        f.write(val_report)
        f.write("\n")
        f.write(test_report)
    print(f"📊 Custom 메트릭 보고서가 저장되었습니다: {report_path}")
