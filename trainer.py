import subprocess
from time import sleep


if __name__ == "__main__":
    dataset_name = input("Enter dataset name (options: CUB-200-2011, FGVC-Aircraft, Stanford-Cars, Iron-Scraps): ").strip()
    
    # 실행할 스크립트 경로 목록 정의
    stages = [
        "trainer_pipeline.racnn_stage1",
        "trainer_pipeline.racnn_stage2",
        "trainer_pipeline.racnn_stage3",
        "trainer_pipeline.racnn_stage4",
        "trainer_pipeline.racnn_stage5",
        "trainer_pipeline.racnn_stage6_final"
    ]
    for i, stage in enumerate(stages):

        print(f"\nStage {i+1} Starting...")
        # 파이썬 모듈(-m) 모드로 실행하여 trainer_pipeline 경로 탐색 문제를 완벽하게 해결합니다.
        result = subprocess.run(args=["uv", "run", "python3", "-m", stage, "--dataset", dataset_name])

        # 에러코드넘버가 0이 아니면, 이전 프로세스에서 학습이 정상적으로 이뤄지지 않은 것으로 판단
        if result.returncode != 0:
            print(f"\n[ERROR] Stage {i+1} failed with exit code {result.returncode}. Aborting training pipeline.")
            break

        print(f"\nStage {i+1} Finished!")
        print("\nWaiting for 3 seconds... to make a .pth file")
        sleep(3)