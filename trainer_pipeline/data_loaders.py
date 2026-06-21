import pandas as pd
import scipy.io
from PIL import Image
from pathlib import Path
from typing import Tuple, Callable

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datasets import load_dataset, DatasetDict


###########################################################
# 데이터셋별 글로벌 경로 변수 (자신의 로컬 환경에 맞게 수정하세요!)
###########################################################

# CUB-200-2011
CUB_DIR = Path("data/benchmark-3/CUB-200-2011")

# FGVC-Aircraft
AIRCRAFT_DIR = Path("data/benchmark-3/FGVC-Aircraft")
AIRCRAFT_IMAGES_DIR = Path(
    "data/benchmark-3/FGVC-Aircraft/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images"
)

# Stanford-Cars
CARS_METADATA_PATH = Path(
    "data/benchmark-3/Stanford-Cars/car_devkit/devkit/cars_meta.mat"
)
CARS_TRAIN_ANNOS_PATH = Path(
    "data/benchmark-3/Stanford-Cars/car_devkit/devkit/cars_train_annos.mat"
)
CARS_TEST_ANNOS_PATH = Path(
    "data/benchmark-3/Stanford-Cars/car_devkit/devkit/cars_test_annos_withlabels.mat"
)
CARS_TRAIN_IMAGES_DIR = Path("data/benchmark-3/Stanford-Cars/cars_train/cars_train")
CARS_TEST_IMAGES_DIR = Path("data/benchmark-3/Stanford-Cars/cars_test/cars_test")
# BogoNet-Iron-Scraps
IRON_SCRAPS_TRAIN_DATASET_DIR = Path(
    "data/Iron-Scraps/set_with_testset/team_share/split_dataset_0pct/train"
)
IRON_SCRAPS_VAL_DATASET_DIR = Path(
    "data/Iron-Scraps/set_with_testset/team_share/split_dataset_0pct/val"
)
IRON_SCRAPS_TEST_DATASET_DIR = Path(
    "data/Iron-Scraps/set_with_testset/team_share/split_dataset_0pct/test"
)


#############################
# CUB-200-2011 데이터셋 클래스
#############################
class Cub_200_2011_Dataset:

    def __init__(self, data_dir: Path = CUB_DIR, download: bool = False) -> None:
        self.data_dir = data_dir
        self.download = download

    def get_raw_dataset(self) -> DatasetDict:
        if self.download:

            print("허깅페이스에서 데이터셋을 다운로드합니다...")

            raw_dataset = load_dataset("bentrevett/caltech-ucsd-birds-200-2011")
            raw_dataset.save_to_disk(self.data_dir)
        else:
            try:
                from datasets import load_from_disk

                raw_dataset = load_from_disk(self.data_dir)

                print(f"[{self.data_dir}] 경로에서 데이터셋을 불러왔습니다.")
            except:

                print(
                    f"[{self.data_dir}] 경로에 데이터가 없습니다. 다운로드를 시작합니다..."
                )

                raw_dataset = load_dataset("bentrevett/caltech-ucsd-birds-200-2011")
                raw_dataset.save_to_disk(self.data_dir)

        return raw_dataset

    def set_transform(self) -> Tuple[Callable, Callable]:
        train_transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def apply_train_transform(example: dict) -> dict:
            example["image"] = [
                train_transform(image.convert("RGB")) for image in example["image"]
            ]

            return example

        def apply_val_transform(example: dict) -> dict:
            example["image"] = [
                val_transform(image.convert("RGB")) for image in example["image"]
            ]

            return example

        return apply_train_transform, apply_val_transform

    def get_dataset(
        self,
        raw_dataset: DatasetDict,
        apply_train_transform: Callable,
        apply_val_transform: Callable,
    ) -> Tuple[Dataset, Dataset]:
        train_dataset = raw_dataset["train"].with_transform(apply_train_transform)
        val_dataset = raw_dataset["test"].with_transform(apply_val_transform)

        return train_dataset, val_dataset


class AircraftDataset(Dataset):

    def __init__(self, csv_file: str, transform):
        super().__init__()
        self.dataframe = pd.read_csv(
            filepath_or_buffer=AIRCRAFT_DIR / csv_file,
            encoding="utf-8",
        )
        self.images_dir = AIRCRAFT_IMAGES_DIR
        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):
        image_name = self.dataframe.iloc[index]["filename"]
        image_path = self.images_dir / image_name

        image = Image.open(fp=image_path)
        label = int(self.dataframe.iloc[index]["Labels"])

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
        }


class CarsDataset(Dataset):

    def __init__(
        self, image_dir: Path, train_or_test: str, transform: transforms = None
    ):
        super().__init__()
        self.transform = transform
        self.image_paths = sorted(image_dir.glob(pattern="*.jpg"))
        self.train_or_test = train_or_test

        if self.train_or_test == "train":
            cars_train_annos = scipy.io.loadmat(file_name=CARS_TRAIN_ANNOS_PATH)
            self.train_labels = [
                int(cars_train_annos["annotations"][0][i][4][0][0]) - 1
                for i in range(len(cars_train_annos["annotations"][0]))
            ]
        elif self.train_or_test == "test":
            cars_test_annos_withlabels = scipy.io.loadmat(
                file_name=CARS_TEST_ANNOS_PATH
            )
            self.test_labels = [
                int(cars_test_annos_withlabels["annotations"][0][i][4][0][0]) - 1
                for i in range(len(cars_test_annos_withlabels["annotations"][0]))
            ]
        else:
            raise ValueError("Unsupported train_or_test value: {self.train_or_test}")

    def __len__(self) -> int:

        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image = Image.open(fp=self.image_paths[index])

        if self.train_or_test == "train":
            label = self.train_labels[index]
        elif self.train_or_test == "test":
            label = self.test_labels[index]
        else:
            raise ValueError(f"Unsupported train_or_test value: {self.train_or_test}")

        if self.transform:
            image = self.transform(image.convert(mode="RGB"))

        return {
            "image": image,
            "label": label,
        }


class DictImageFolder(ImageFolder):
    """ImageFolder의 반환 타입을 다른 데이터셋과 일치하도록 딕셔너리로 변환하는 래퍼 클래스"""

    def __getitem__(self, index: int) -> dict:
        image, label = super().__getitem__(index)
        return {
            "image": image,
            "label": label,
        }


######################
# 보고넷 철스크랩 데이터
######################
def get_iron_scraps_dataset(
    train_dataset_dir: Path, val_dataset_dir: Path
) -> tuple[Dataset, Dataset]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = DictImageFolder(
        root=train_dataset_dir,
        transform=train_transform,
    )
    val_dataset = DictImageFolder(
        root=val_dataset_dir,
        transform=val_transform,
    )

    return train_dataset, val_dataset


def get_dataloaders(
    dataset_name: str, batch_size: int = 16, download: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Args:
        dataset_name (str): 데이터셋 이름 ("CUB-200-2011", "FGVC-Aircraft", "Stanford-Cars", "Iron-Scraps")
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, val_loader_shuffle)
    """
    if dataset_name == "CUB-200-2011":
        cub_200_2011_dataset = Cub_200_2011_Dataset(data_dir=CUB_DIR, download=download)
        apply_train_transform, apply_val_transform = (
            cub_200_2011_dataset.set_transform()
        )
        raw_dataset = cub_200_2011_dataset.get_raw_dataset()

        train_dataset, val_dataset = cub_200_2011_dataset.get_dataset(
            raw_dataset, apply_train_transform, apply_val_transform
        )
    elif dataset_name == "FGVC-Aircraft":
        train_transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = AircraftDataset(csv_file="train.csv", transform=train_transform)
        val_dataset = AircraftDataset(csv_file="test.csv", transform=val_transform)
    elif dataset_name == "Stanford-Cars":
        train_transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = CarsDataset(
            image_dir=CARS_TRAIN_IMAGES_DIR,
            train_or_test="train",
            transform=train_transform,
        )
        val_dataset = CarsDataset(
            image_dir=CARS_TEST_IMAGES_DIR,
            train_or_test="test",
            transform=val_transform,
        )
    elif dataset_name == "Iron-Scraps":
        train_dataset, val_dataset = get_iron_scraps_dataset(
            train_dataset_dir=IRON_SCRAPS_TRAIN_DATASET_DIR,
            val_dataset_dir=IRON_SCRAPS_VAL_DATASET_DIR,
        )
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )
    val_loader_shuffle = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    print(f"학습용 데이터 샘플 수: {len(train_dataset)}")
    print(f"검증용 데이터 샘플 수: {len(val_dataset)}")
    print(f"학습용 배치 수: {len(train_loader)}")
    print(f"검증용 배치 수: {len(val_loader)}")
    print(f"검증용 셔플 배치 수: {len(val_loader_shuffle)}")

    return train_loader, val_loader, val_loader_shuffle


def get_num_classes(dataset_name: str) -> int:
    """데이터셋 이름에 따른 클래스 개수 반환"""
    dataset_map = {
        "CUB-200-2011": 200,
        "FGVC-Aircraft": 100,
        "Stanford-Cars": 196,
        "Iron-Scraps": 3,
    }
    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    return dataset_map[dataset_name]


def get_test_dataloader(
    dataset_name: str, batch_size: int = 16
) -> DataLoader:
    """테스트용 데이터로더를 반환합니다."""
    if dataset_name == "Iron-Scraps":
        test_transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        test_dataset = DictImageFolder(
            root=IRON_SCRAPS_TEST_DATASET_DIR,
            transform=test_transform,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
        )
        print(f"테스트용 데이터 샘플 수: {len(test_dataset)}")
        print(f"테스트용 배치 수: {len(test_loader)}")
        return test_loader
    else:
        raise NotImplementedError(f"Test dataloader not implemented for {dataset_name}")
