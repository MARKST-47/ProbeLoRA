from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from typing import Tuple

# Normalisation constants
NORM_STATS = {
    "imagenet": {
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
    },
    "clip": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std":  [0.26862954, 0.26130258, 0.27577711],
    },
}

def get_transform(backbone_norm: str, split: str) -> transforms.Compose:
    """
    Creates standard image transform pipelines.
    Both ViT backbones expect 224x224 RGB inputs.
    """
    stats = NORM_STATS[backbone_norm]
    normalize = transforms.Normalize(mean=stats["mean"], std=stats["std"])
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:  
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

class CUB200(Dataset):
    """
    Caltech-UCSD Birds 200-2011 dataset class.
    Parses structural text mappings directly from raw data files.
    """
    def __init__(self, root: str, train: bool = True, transform=None):
        self.root = Path(root).resolve()
        # Fallback check for nested archive extraction paths
        if (self.root / "CUB_200_2011").exists():
            self.root = self.root / "CUB_200_2011"
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(
                f"CUB-200 data structure not found at {self.root}. Check download paths."
            )
        # To parse structural index pairs
        def read_pairs(filename: str):
            pairs = {}
            with open(self.root / filename, "r") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if parts:
                        pairs[int(parts[0])] = parts[1]
            return pairs
        img_paths = read_pairs("images.txt")
        img_labels = {k: int(v) - 1 for k, v in read_pairs("image_class_labels.txt").items()}
        is_train_split = {k: int(v) for k, v in read_pairs("train_test_split.txt").items()}
        target_flag = 1 if train else 0
        self.samples = [
            (str(self.root / "images" / img_paths[i]), img_labels[i])
            for i in img_paths if is_train_split[i] == target_flag
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloader(
    dataset_name: str,
    backbone_norm: str,
    split: str,
    data_root: str = "./data/raw",
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, int]:
    """
    Unified DataLoader generation factory for the experiment matrix.
    """
    transform = get_transform(backbone_norm, split)
    root = Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if dataset_name == "cifar100":
        ds = datasets.CIFAR100(
            root=str(root / "cifar100"),
            train=(split == "train"),
            download=True,
            transform=transform,
        )
        num_classes = 100   
    elif dataset_name == "oxford_pets":
        tv_split = "trainval" if split == "train" else "test"
        ds = datasets.OxfordIIITPet(
            root=str(root / "oxford_pets"),
            split=tv_split,
            target_types="category",
            download=True,
            transform=transform,
        )
        num_classes = 37
    elif dataset_name == "cub200":
        ds = CUB200(
            root=str(root / "cub200"),
            train=(split == "train"),
            transform=transform,
        )
        num_classes = 200
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader, num_classes
