import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from trainer_pipeline.data_loaders import get_num_classes
from trainer_pipeline.model_base_architectures.ra_cnn import RACNN, prepare_modules
from trainer_pipeline.gradcam import visualize_and_save_gradcam_comparison

def generate_presentation_heatmaps(dataset_name: str = "Iron-Scraps"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name)
    
    # 모델 모듈 로드
    classifier1, classifier2, classifier3, apn1, apn2 = prepare_modules(
        classifier1_path=f"models/{dataset_name}/{dataset_name}_best_on_stage1.pth",
        apn1_path=f"models/{dataset_name}/{dataset_name}_best_on_stage2.pth",
        classifier2_path=f"models/{dataset_name}/{dataset_name}_best_on_stage3.pth",
        apn2_path=f"models/{dataset_name}/{dataset_name}_best_on_stage4.pth",
        classifier3_path=f"models/{dataset_name}/{dataset_name}_best_on_stage5.pth",
        num_classes=num_classes,
    )

    model = RACNN(
        classifier1=classifier1,
        classifier2=classifier2,
        classifier3=classifier3,
        apn1=apn1,
        apn2=apn2,
    ).to(device)

    # Best stage 6 weights
    best_model_path = Path(f"models/{dataset_name}/{dataset_name}_best_on_stage6.pth")
    if not best_model_path.exists():
        print(f"[오류] 모델 가중치 파일이 존재하지 않습니다: {best_model_path}")
        print("먼저 모델 학습(stage6_final)을 완료해주세요.")
        return
        
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    # 이미지 불러오기
    samples_dir = Path("data/Iron-Scraps/samples_for_presentation")
    image_paths = sorted(list(samples_dir.glob("*.jpg")))
    
    if not image_paths:
        print(f"[오류] 프리젠테이션용 샘플 이미지가 존재하지 않습니다: {samples_dir}")
        return

    print(f"총 {len(image_paths)}개의 샘플에 대해 히트맵을 생성합니다...")

    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images_list = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images_list.append(transform(img))
        
    images_tensor = torch.stack(images_list).to(device)
    
    # Inference
    print("모델 추론 중...")
    with torch.no_grad():
        logits1, logits2, logits3, crop1, crop2 = model(images_tensor)
        ensemble_logits = (logits1 + logits2 + logits3) / 3.0
        _, predictions = ensemble_logits.max(dim=1)

    # 클래스 이름 (알파벳 순서 가정)
    class_names = ["cut", "danger", "excluded"]
    
    # Ground Truth가 파일명에 있는지 유추, 없으면 Prediction과 동일하게 설정하여 O/X 렌더링 우회
    labels_list = []
    for i, p in enumerate(image_paths):
        name = p.name.lower()
        if "danger" in name:
            labels_list.append(1)
        elif "cut" in name:
            labels_list.append(0)
        elif "excluded" in name:
            labels_list.append(2)
        else:
            # 파일명에 레이블 정보가 없으면 모델의 예측을 그대로 GT로 간주
            labels_list.append(predictions[i].item())
            
    labels_tensor = torch.tensor(labels_list).to(device)

    # 히트맵 저장
    print("Grad-CAM++ 히트맵 생성 중... (시간이 다소 소요될 수 있습니다)")
    visualize_and_save_gradcam_comparison(
        dataset_name=f"{dataset_name}_presentation",  # 별도의 파일명으로 저장하기 위함
        images=images_tensor,
        crop1=crop1,
        crop2=crop2,
        predictions=predictions,
        labels=labels_tensor,
        class_names=class_names,
        model=model,
        device=device,
    )
    
    print("완료되었습니다. 결과는 results/Iron-Scraps_presentation/gradcam_comparison.png 에서 확인하실 수 있습니다.")

if __name__ == "__main__":
    generate_presentation_heatmaps()
