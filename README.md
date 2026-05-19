# RA-CNN (Recurrent Attention Convolutional Neural Network)

이 프로젝트는 **Fine-grained Visual Classification (FGVC)** 을 위한 Attention 기반 Convolutional Neural Network (CNN) 아키텍처를 구현한 코드입니다.  
본 구현은 논문 **"Learning Deep Features for Discriminative Localization (ICLR 2018)"** 의 핵심 메커니즘을 기반으로 하고 있습니다.


## 핵심 모듈 구성
본 아키텍처는 세 가지 스케일(Scale)을 유기적으로 제어하기 위해 총 5개의 모듈로 구성됩니다.

1. **`classifier1`**: Scale 1의 기본 이미지 분류기 (Vanilla CNN)
2. **`apn1`**: Scale 1 Feature Map에서 핵심 영역을 감지하여 Scale 2로 크롭 및 스케일링하는 Attention Proposal Network
3. **`classifier2`**: Scale 2(크롭된 국소 영역)의 분류기
4. **`apn2`**: Scale 2 Feature Map에서 더욱 미세한 영역을 감지하여 Scale 3로 크롭 및 스케일링하는 APN
5. **`classifier3`**: Scale 3(더욱 미세한 국소 영역)의 분류기


## 학습 및 앙상블 전략

* **단계별 독립 학습 (Stage-wise Training)**: 
  모듈은 한 번에 하나씩 순차적으로 학습(Stage 1 ~ Stage 5)되며, 특정 모듈이 학습되는 동안 나머지 모듈의 가중치는 완전히 동결(Freezing)됩니다.

* **최종 정렬 및 공동 최적화 (Stage 6)**: 
  마지막 Stage 6에서는 모든 모듈의 동결을 해제하고 전체 네트워크를 동시에 미세 조정(Fine-tuning)합니다.

* **단순 합 기반 앙상블 (Ensemble)**: 
  세 가지 스케일의 분류 로짓(Logits)을 합산하여 최종 예측을 수행하는 간소화된 앙상블 로직을 채택하였습니다.


## 디렉토리 구조

```text
├── README.md
├── data
│   └── benchmark-3
│       ├── CUB-200-2011
│       ├── FGVC-Aircraft
│       └── Stanford-Cars
├── hyper-parameters.yaml
├── simple_tester.py    
├── trainer.py
└── trainer_pipeline
    ├── data_loaders.py
    ├── model_base_architectures/
    ├── racnn_stage1.py
    ├── racnn_stage2.py
    ├── racnn_stage3.py
    ├── racnn_stage4.py
    ├── racnn_stage5.py
    └── racnn_stage6_final.py
```


## 사용법

### 0. 환경 설정 및 의존성 설치

프로젝트 루트 폴더에서 `uv`를 활용해 가상환경 및 의존성을 정렬합니다.
```bash
uv sync
```

### 1. 모델 학습 (`trainer.py`)

각 스테이지(Stage 1 ~ 6)를 순차적으로 실행하여 학습을 진행하고, 완성된 최적 가중치를 `./models/{dataset_name}/` 경로에 차례대로 저장합니다.
* 하이퍼파라미터는 `hyper-parameters.yaml` 파일에서 통합 관리됩니다.
* 실행 시 대화형 입력 창이 뜨며 학습하고자 하는 데이터셋 명을 입력해야 합니다.
> [!NOTE]
> 데이터 경로 관련 오류가 발생할 경우, `trainer_pipeline/data_loaders.py` 내의 경로 설정을 로컬 환경에 맞게 확인해 주시기 바랍니다.

* 실행 코드
```bash
uv run python3 -m trainer
```


### 2. 시각적 테스트 및 검증 (`simple_tester.py`)

학습이 완료된 가중치를 불러와 테스트 데이터셋의 예측 지표(Accuracy, Precision, Recall, F1)를 출력하고, 랜덤으로 선택된 10개의 샘플에 대한 스케일별 크롭 이미지 결과를 시각적으로 저장합니다.
* 결과물(지표 및 크롭 이미지)은 `results/{dataset_name}/` 디렉토리에 저장됩니다.

* 실행 코드
```bash
uv run python3 simple_tester.py
```

## results

* 측정 지표
<img width="1500" height="1000" alt="stage6_CUB-200-2011_learning_curves" src="https://github.com/user-attachments/assets/4f3c8ab5-1c9e-467b-97fd-20567962866f" />

* full_image
<img width="1692" height="678" alt="scale1_full" src="https://github.com/user-attachments/assets/6b2034ed-035a-42be-a418-210a7cea0304" />

* scale1_image
<img width="1692" height="678" alt="scale2_crop" src="https://github.com/user-attachments/assets/be80999e-d458-4c9e-9431-9c47d3fa9961" />

* scale2_image
<img width="1692" height="678" alt="scale3_crop" src="https://github.com/user-attachments/assets/45198f64-6384-48b0-8f8e-28a3e2756257" />





