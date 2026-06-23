import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from typing import Tuple


# Normalisation constants
# DINOv2 was trained with ImageNet statistics.
# CLIP has its own distinct normalisation.
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
    Both ViT backbones expect 224×224 RGB images.
    Training split uses random crop + horizontal flip for light augmentation
    during the probing stage.

    Args:
        backbone_norm: one of "imagenet" | "clip"
        split: "train" | "val"
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


# CUB-200-2011 custom dataset 
class CUB200(Dataset):
    """
    Caltech-UCSD Birds 200-2011.

    Expects the following directory layout after running
    scripts/download_cub200.sh:

        <root>/CUB_200_2011/
            images.txt              (image_id → relative/path.jpg)
            image_class_labels.txt  (image_id → class_id, 1-indexed)
            train_test_split.txt    (image_id → 1=train / 0=test)
            images/
                001.Black_footed_Albatross/
                    Black_Footed_Albatross_0001_796111.jpg
                    ...
    """
    def __init__(self, root: str, train: bool = True,
                 transform=None):
        self.root = Path(root) / "CUB_200_2011"
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"CUB-200 not found at {self.root}. "
                "Run scripts/download_cub200.sh first."
            )

        # Parse the three index files
        def read_pairs(fname):
            with open(self.root / fname) as f:
                return {int(a): b for line in f
                        for a, b in [line.strip().split(maxsplit=1)]}

        img_paths = read_pairs("images.txt")                       # id → path
        img_labels = {k: int(v) - 1 for k, v in read_pairs("image_class_labels.txt").items()}
        is_train = {k: int(v) for k, v in read_pairs("train_test_split.txt").items()}

        flag = 1 if train else 0
        self.samples = [
            (str(self.root / "images" / img_paths[i]), img_labels[i])
            for i in img_paths
            if is_train[i] == flag
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# Public factory
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
    Returns (DataLoader, num_classes) for the requested dataset/split.

    Args:
        dataset_name:  "cifar100" | "oxford_pets" | "cub200"
        backbone_norm: "imagenet" | "clip"  — controls normalisation
        split:         "train" | "val"
        data_root:     directory where raw datasets are stored / downloaded
        batch_size:    images per batch
        num_workers:   DataLoader worker processes (4 for local, 8 on cluster)
        pin_memory:    set True when using GPU

    Returns:
        loader      : torch.utils.data.DataLoader
        num_classes : int
    """
    transform = get_transform(backbone_norm, split)
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    # CIFAR-100
    if dataset_name == "cifar100":
        is_train = (split == "train")
        ds = datasets.CIFAR100(
            root = str(root / "cifar100"),
            train = is_train,
            download = True,
            transform = transform,
        )
        num_classes = 100

    # Oxford-IIIT Pets
    elif dataset_name == "oxford_pets":
        # torchvision splits: "trainval" (3680 imgs) or "test" (3669 imgs)
        tv_split = "trainval" if split == "train" else "test"
        ds = datasets.OxfordIIITPet(
            root = str(root / "oxford_pets"),
            split = tv_split,
            target_types = "category",
            download = True,
            transform = transform,
        )
        num_classes = 37

    # CUB-200-2011 Birds
    elif dataset_name == "cub200":
        is_train = (split == "train")
        ds = CUB200(
            root = str(root / "cub200"),
            train = is_train,
            transform = transform,
        )
        num_classes = 200

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            "Choose from: cifar100, oxford_pets, cub200"
        )

    loader = DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = (split == "train"),
        num_workers = num_workers,
        pin_memory = pin_memory,
        drop_last = False,
    )

    return loader, num_classes


# Quick check 
if __name__ == "__main__":
    print("Testing dataloaders (CIFAR-100 only — downloads ~160 MB)…")
    for norm in ["imagenet", "clip"]:
        for split in ["train", "val"]:
            loader, n = get_dataloader("cifar100", norm, split,
                                       data_root="./data/raw",
                                       batch_size=64, num_workers=0)
            x, y = next(iter(loader))
            print(f"cifar100 | norm={norm} | split={split} | "
                  f"batch={list(x.shape)} | classes={n}")
    print("All OK.")
