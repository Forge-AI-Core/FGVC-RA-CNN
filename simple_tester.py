import argparse
from pathlib import Path
import torchvision.utils as vutils

import yaml
import torch
from torch import nn

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from trainer_pipeline.model_base_architectures.vanilla_cnn import VanillaCNN
from trainer_pipeline.model_base_architectures.apn import APN
from trainer_pipeline.racnn_stage6_final import Stage6Model
from trainer_pipeline.data_loaders import get_dataloaders, get_num_classes


class FinalModel(nn.Module):
    """최종 앙상블 모델"""
    def __init__(self, model_path: Path, num_classes: int):
        super().__init__()
        
        # 속이 텅 빈 기본 모듈 5개를 만듭니다. (데이터셋에 부합하는 클래스 개수로 동적 설정)
        classifier1 = VanillaCNN(num_classes=num_classes)
        classifier2 = VanillaCNN(num_classes=num_classes)
        classifier3 = VanillaCNN(num_classes=num_classes)
        apn1 = APN(in_features=512*10*10)
        apn2 = APN(in_features=512*10*10)
        
        # 이 5개를 조립하여 빈 Stage6Model 구조체를 만듭니다.
        self.racnn = Stage6Model(classifier1, classifier2, classifier3, apn1, apn2)
        
        # 완성된 뼈대에 학습시킨 최종 가중치를 덮어씌웁니다.
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        self.racnn.load_state_dict(state_dict)

    def forward(self, x):
        # 세 가지 시야(Scale 1, 2, 3)의 결과를 모두 가져옵니다.
        logits1, logits2, logits3, crop_image1, crop_image2 = self.racnn(x)
        
        # 앙상블(세 시야의 종합 의견)을 계산하여 최종 결과로 반환합니다.
        ensemble_logits = logits1 + logits2 + logits3
        
        return ensemble_logits, crop_image1, crop_image2


def get_model(model_path: Path, num_classes: int) -> nn.Module:
    """최종 앙상블 모델을 불러오는 함수"""
    model = FinalModel(model_path=model_path, num_classes=num_classes)

    return model


def test_final_model(model, test_loader, criterion, device, dataset_name):
    """최종 앙상블 모델의 테스트 코드 (전체 평가지표 계산)"""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        progress_bar = tqdm(iterable=test_loader, desc="전체 평가 중")
        for index, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _, _ = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item()
            total += labels.size(dim=0)
            _, predictions = logits.max(dim=1)
            correct += predictions.eq(labels).sum().item()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            avg_loss = running_loss / (index + 1)
            acc = (correct / total) * 100.0
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.2f}%"})
        
    # 최종 지표 계산
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # 클래스명 추출
    if hasattr(test_loader.dataset, 'classes'):
        class_names = test_loader.dataset.classes
    else:
        import numpy as np
        num_classes = len(np.unique(all_labels))
        class_names = [f"Class {i}" for i in range(num_classes)]

    # 1. classification report (클래스별 프리시전, 리콜, 종합 어큐러시 포함)
    report_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

    print("\n" + "="*60)
    print("              [최종 테스트 셋 상세 평가 보고서]")
    print("="*60)
    print(report_str)
    print("="*60 + "\n")

    # 결과 폴더 생성 및 저장
    results_dir = Path(f"results/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "test_metrics_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f" 데이터셋: {dataset_name}\n")
        f.write("="*60 + "\n\n")
        f.write(report_str)
    print(f"📊 상세 메트릭 보고서가 저장되었습니다: {report_path}")

    # 2. Confusion Matrix 계산 및 시각화
    cm = confusion_matrix(all_labels, all_preds)
    import numpy as np

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({dataset_name})", fontsize=16)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    cm_path = results_dir / "test_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"📸 혼동 행렬 시각화 이미지가 저장되었습니다: {cm_path}\n")
    
    return avg_loss, acc, precision, recall, f1


def load_hyper_parameters(hyperparameter_path: Path) -> dict:
    """하이퍼 파라미터를 불러오는 코드"""
    with open(hyperparameter_path, "r", encoding="utf-8") as file:
        hyperparameters = yaml.safe_load(file)

    return hyperparameters


def main(dataset_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 모델 정상적으로 불러오기 및 GPU 할당 (새로운 중첩 폴더 구조 적용)
    model_path = Path(f"models/{dataset_name}/{dataset_name}_best_on_stage6.pth")
    if not model_path.exists():
        print(f"모델 가중치 파일({model_path})이 없습니다. Stage 6 학습이 완료되었는지 확인하세요.")
        return

    print("모델을 불러오는 중...")
    num_classes = get_num_classes(dataset_name)
    model = get_model(model_path, num_classes).to(device)
    model.eval()

    # 2. 전체 데이터셋 평가
    print("\n[1단계] 전체 테스트셋 평가를 시작합니다...")
    # 평가 시에는 속도를 위해 배치를 좀 더 크게 잡습니다.
    _, test_loader, _ = get_dataloaders(dataset_name=dataset_name, batch_size=32)
    criterion = nn.CrossEntropyLoss()
    
    _, acc, p, r, f1 = test_final_model(model, test_loader, criterion, device, dataset_name)
    
    print("\n" + "="*50)
    print(f"       {dataset_name.upper()} 전체 테스트셋 최종 평가 결과 (Ensemble)")
    print("="*50)
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*50 + "\n")

    # 3. 딱 10장짜리 배치를 만들기 위해 batch_size=10 으로 데이터로더 호출
    print("[2단계] 랜덤 10장 샘플 테스트를 시작합니다...")
    _, _, test_loader_shuffle = get_dataloaders(dataset_name=dataset_name, batch_size=10)

    # 셔플 로더에서 딱 1개의 배치(10장)만 뽑아냄
    batch = next(iter(test_loader_shuffle))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    # 추론
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            ensemble_logits, crop1, crop2 = model(images)

    # 상세 결과 출력
    _, predictions = ensemble_logits.max(dim=1)
    
    print("="*50)
    print(f"{'No.':<5} | {'정답(GT)':<8} | {'예측(Pred)':<10} | {'결과':<5}")
    print("-" * 50)
    for i in range(len(labels)):
        gt = labels[i].item()
        pred = predictions[i].item()
        is_correct = "O" if gt == pred else "X"
        print(f"{i+1:<5} | {gt:<8} | {pred:<10} | {is_correct:<5}")
    print("-" * 50)
    print("📸 이미지가 저장되었습니다: scale1_full.png / scale2_crop.png / scale3_crop.png")
    print("="*50 + "\n")

    # 저장용 결과 디렉토리 생성
    Path(f"results/{dataset_name}").mkdir(parents=True, exist_ok=True)

    # Tensor를 실제 이미지 파일(.png)로 저장하여 눈으로 확인
    vutils.save_image(images, f"results/{dataset_name}/scale1_full.png", normalize=True, nrow=5)
    vutils.save_image(crop1, f"results/{dataset_name}/scale2_crop.png", normalize=True, nrow=5)
    vutils.save_image(crop2, f"results/{dataset_name}/scale3_crop.png", normalize=True, nrow=5)


if __name__ == "__main__":
    dataset_name = input("데이터셋 이름을 입력해주세요(FGVC-Aircraft / Stanford-Cars / CUB-200-2011 / Iron-Scraps): ").strip()
    if dataset_name not in ["FGVC-Aircraft", "Stanford-Cars", "CUB-200-2011", "Iron-Scraps"]:
        raise ValueError("잘못된 데이터셋 이름입니다. 다시 입력해주세요.")

    main(dataset_name=dataset_name)