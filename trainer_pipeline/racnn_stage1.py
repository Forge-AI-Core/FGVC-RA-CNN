from pathlib import Path

from torch.utils.data import DataLoader

from typing import Any
import yaml
import argparse

import torch
from torch import nn, optim
from tqdm import tqdm

from trainer_pipeline.data_loaders import get_dataloaders, get_num_classes
from trainer_pipeline.model_base_architectures.vanilla_cnn import VanillaCNN


####################################################################
# Stage1에서는 crop 로직 없이 오직 Scale1 이미지로만 분류기를 학습시킵니다.
####################################################################


####################
# Stage1의 학습 코드
####################
def train_stage1(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str) -> float:
    """stage1의 학습 코드"""
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
            logits = model(images)
            classification_loss = criterion(logits, labels)

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
# Stage1의 테스트 코드
#####################
def test_stage1(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: str) -> float:
    """stage1의 테스트 코드"""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    # with torch.no_grad(): 가장 외곽에 위치
    with torch.no_grad():
        progress_bar = tqdm(iterable=test_loader, desc="TEST")
        for index, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(images)
                classification_loss = criterion(logits, labels)
                        
            # for loss
            running_loss += classification_loss.item()
            avg_loss = running_loss / (index + 1)

            # for accuracy
            total += labels.size(dim=0)
            _, predictions = logits.max(dim=1)
            correct += predictions.eq(labels).sum().item()
            acc = correct / total * 100.0

            # set.postfix: 프린트를 일일이 출력하는 대신, 진행바에 표시하여 화면이 길게 늘어지는 것을 방지
            # 프로그레스 바 뒤에 고정되는 것은 인자로 입력한 딕셔너리
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{acc:.2f}%"
            })

    return acc


####################
# 하이퍼 파라미터 로드
####################
def load_hyper_parameters(hyperparameter_path: Path) -> dict[str, Any]:
    """하이퍼 파라미터를 불러오는 코드"""
    with open(hyperparameter_path, "r", encoding="utf-8") as file:
        hyperparameters = yaml.safe_load(file)

    return hyperparameters


##############################
# 학습된 Classifier1을 얻는 함수
##############################
def get_classifier1(dataset_name: str, hyperparameter_path: Path) -> None:
    """classifier1의 main 코드"""
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

    # 모델
    model = VanillaCNN(num_classes=num_classes).to(device)
    
    # 옵티마이저, 손실함수, 학습 속도 스케줄러
    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        train_stage1(model, train_loader, criterion, optimizer, device)
        acc = test_stage1(model, test_loader, criterion, device)

        # 스케줄러의 스텝은 모델 내부가 아닌 루프 내부에서 수행
        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            Path(f"models/{dataset_name}").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"models/{dataset_name}/{dataset_name}_best_on_stage1.pth")

            print(f"best_acc: {best_acc:.2f}%")
        else:
            patience_counter += 1

            print(f"EarlyStopping patience counter: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:

                print("Early Stopping triggered.")
                print(f"final_acc: {best_acc:.2f}%")
                break


if __name__ == "__main__":
    # 터미널 또는 서브프로세스 라이브러리가 호출명령어를 전달했을 때 이를 받는 담당자는 이 상황에서는 2 가지로 구분됩니다.
    # 첫 번째는 파이썬 인터프리터이고, 두 번쨰는 파서 인스턴스입니다.
    # 파이썬 인터프리터는 .py까지만 보고 실행하며 호출 명령어는 sys.argv라는 주머니에 얌전히 보관되어있습니다.
    # 만약, 파서가 없다면 등록되지 않은 옵션 또는 플래그가 전달되더라도 인터프리터는 .py까지만 확인하므로 \
    # 아무 에러도 발생하지 않습니다.
    # 파서 인스턴스가 있다고 하더라도 담당자만 생서된 것에 불과하므로, 여전히 에러는 발생하지 않습니다.
    # 파서 인스턴스가 .add_argument라는 메서드를 통해 플래그를 등록한 경우에만, sys.argv를 보고 플래그가 있는지 검사합니다.
    # 그러나 이 경우에도 여전히 플래그가 일치한다면, 그 내용이 무엇이든 상관없이 에러는 발생하지 않고 \
    # 정상적으로 .py를 실행합니다.
    # 파서가 parse_args() 메서드를 실행시켜 args를 생성했다면 sys.argv의 내용이 args.인자명의 형태로 args에 저장되고 \
    # 사용자가 그 객체를 사용할 수 있게 됩니다.
    
    # 파서 인스턴스를 생성
    parser = argparse.ArgumentParser(description="Options: CUB-200-2011, FGVC-Aircraft, Stanford-Cars")
    # 인자(플래그 또는 옵션 이름) 등록
    parser.add_argument("--dataset", type=str, required=True, help="데이터셋 이름을 입력하세요.")
    parser.add_argument("--hyperparameter", type=str, default="hyper-parameters.yaml", help="YAML파일 경로")
    # sys.argv를 파싱하여 args를 생성하고 이 객체에 인자값을 저장합니다.
    args = parser.parse_args()

    # args의 dataset 어트리뷰트를 인자로 전달하여 메인 로직을 호출합니다.
    get_classifier1(
        dataset_name=args.dataset,
        hyperparameter_path=Path(args.hyperparameter),
    )