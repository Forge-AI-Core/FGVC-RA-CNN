from pathlib import Path
import torch
from torch import nn

from trainer_pipeline.model_base_architectures.vanilla_cnn import VanillaCNN
from trainer_pipeline.model_base_architectures.apn import APN, crop_image


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
    cls1_state_dict = torch.load(
        classifier1_path, map_location="cpu", weights_only=True
    )
    cls2_state_dict = torch.load(
        classifier2_path, map_location="cpu", weights_only=True
    )
    cls3_state_dict = torch.load(
        classifier3_path, map_location="cpu", weights_only=True
    )
    apn1_state_dict = torch.load(apn1_path, map_location="cpu", weights_only=True)
    apn2_state_dict = torch.load(apn2_path, map_location="cpu", weights_only=True)

    # 분류기 1
    classifier1 = VanillaCNN(num_classes=num_classes)
    classifier1.load_state_dict(state_dict=cls1_state_dict)

    # 분류기 2
    classifier2 = VanillaCNN(num_classes=num_classes)
    # 모델객체는 stage2의 Stage2Model 클래스(classifier와 apn을 포함)입니다.
    # 따라서 스페시픽한 classfier를 먼저 추출해주고 추출한 classifier2의 가중치에서 "classifier2." 접두어를 제거합니다.
    c2_weights = {
        k.replace("classifier2.", ""): v
        for k, v in cls2_state_dict.items()
        if k.startswith("classifier2.")
    }
    classifier2.load_state_dict(state_dict=c2_weights)

    # 분류기 3
    classifier3 = VanillaCNN(num_classes=num_classes)
    # 가중치 딕셔너리에서 "classifier3." 접두어를 제거하여 로드합니다.
    c3_weights = {
        k.replace("classifier3.", ""): v
        for k, v in cls3_state_dict.items()
        if k.startswith("classifier3.")
    }
    classifier3.load_state_dict(state_dict=c3_weights)

    # APN 1
    apn1 = APN(in_features=512 * 7 * 7)
    # 파이토치의 state_dict는 {"레이어 이름": "가중치 텐서"} 형태의 파이썬 딕셔너리입니다.
    # 키와 밸류는 각각 다음을 의미합니다:
    # 키: 레이어의 이름 (예: 'apn.fc1.weight', 'apn.fc2.bias')
    # 밸류: 해당 레이어에 저장된 실제 가중치 숫자들(Tensor)
    apn1_weights = {
        k.replace("apn.", ""): v
        for k, v in apn1_state_dict.items()
        if k.startswith("apn.")
    }
    apn1.load_state_dict(apn1_weights)

    # APN 2
    apn2 = APN(in_features=512 * 7 * 7)
    # 가중치 딕셔너리에서 "apn2." 접두어를 제거하여 로드합니다.
    apn2_weights = {
        k.replace("apn2.", ""): v
        for k, v in apn2_state_dict.items()
        if k.startswith("apn2.")
    }
    apn2.load_state_dict(apn2_weights)

    return classifier1, classifier2, classifier3, apn1, apn2


class RACNN(nn.Module):

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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
