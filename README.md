# RACNN-pytorch
이 프로젝트는 RA-CNN(Recurrent Attention Convolutional Neural Network)을 PyTorch로 구현한 서드파티 저장소입니다. 현재 [논문](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/07/Look-Closer-to-See-Better-Recurrent-Attention-Convolutional-Neural-Network-for-Fine-grained-Image-Recognition.pdf)에 명시된 성능을 재현하기 위해 작업 중입니다.

CUB200 데이터셋은 [이 페이지](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)에서 다운로드할 수 있으며, 다음 명령어로 `data/` 폴더에 압축을 풀 수 있습니다: `tar -xvf CUB_200_2011.tgz -C data/`

## 요구 사항
- [uv](https://github.com/astral-sh/uv) (매우 빠른 Python 패키지 관리자)
- Python 3.12+ (uv를 통해 관리됨)

## 설정 및 학습 방법
1. 이 저장소를 클론합니다.
2. Hugging Face `datasets` 라이브러리를 사용하여 CUB-200-2011 데이터셋을 다운로드합니다. 데이터셋은 `./data/cub-200-2011` 경로에 Arrow 형식으로 위치해야 합니다.
3. `uv`를 사용하여 의존성을 설치하고 학습을 실행합니다:
```bash
uv sync
uv run python trainer.py
```

## 진행 상황 (TODO)
- [x] 네트워크 구축
- [ ] 인자(Arguments) 리팩토링
- [ ] APN 사전 학습 구현
- [ ] APN과 ConvNet/Classifier 간의 교차 학습 구현
- [ ] 성능 재현 및 README.md 업데이트
- [ ] 샘플 시각화
  - [이 구현체](https://github.com/klrc/RACNN-pytorch)를 참고함
- [ ] 성능 향상을 위한 새로운 접근 방식 추가

## 현재 이슈
- APN 사전 학습에 대한 구체적인 디테일 부족
- Rankloss가 감소하지 않음 (사전 학습의 부재 또는 버그 가능성)

## 실험 결과
현재 최고 성능은 APN 사전 학습 없이 Scale 1에서 **71.68%**입니다. 이는 일반적인 VGG19만 사용하는 것보다 낮은 수치입니다.

## 사용법
학습을 위해 다음 명령어를 사용하세요:
```bash
$ uv run python trainer.py
```
또는
```bash
$ uv run bash train.sh
```

현재 CUDA가 가능한 디바이스만 지원합니다.

학습 과정을 모니터링하려면 다음을 실행하세요:
```bash
$ uv run tensorboard --logdir='visual/' --port=6666
```
그 후 웹 브라우저에서 `localhost:6666`에 접속하여 Loss, Accuracy 등을 확인할 수 있습니다.

## 참고 문헌
- [원문 코드 (Caffe)](https://github.com/Jianlong-Fu/Recurrent-Attention-CNN)
- [다른 PyTorch 구현체](https://github.com/Charleo85/DeepCar) (Attention Crop 코드를 참고함)
