"""dataset.py — Quản lý dữ liệu Camelyon17-WILDS.

Tải dataset, tạo các split và xây dựng DataLoader.
"""

import ssl
import urllib.request

# ponytail: bypass SSL cho CodaLab expired cert, bỏ khi server fix SSL
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from wilds import get_dataset


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform():
    """Pipeline đánh giá cố định: resize, tensor, normalize."""
    return transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_transform(use_augmentation=False):
    """Pipeline training: baseline hoặc augmentation."""
    if not use_augmentation:
        return get_eval_transform()
    return transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(batch_size=16, use_augmentation=False, data_dir="data", num_workers=0):
    """Tải Camelyon17-WILDS và trả về DataLoader cho các split.

    Returns:
        dict[str, DataLoader]: dataloaders cho 'train', 'id_val', 'val', 'test'
        int: hospital_idx — vị trí metadata bệnh viện
        int: n_groups — số lượng nhóm (bệnh viện)
    """
    dataset = get_dataset(dataset="camelyon17", root_dir=data_dir, download=True)

    splits = {}
    split_names = {
        "train": "train",
        "id_val": "id_val",
        "val": "val",
        "test": "test",
    }

    train_transform = get_train_transform(use_augmentation)
    eval_transform = get_eval_transform()

    for key, split_name in split_names.items():
        subset = dataset.get_subset(split_name,
                                    transform=train_transform if key == "train" else eval_transform)
        splits[key] = DataLoader(subset,
                                 batch_size=batch_size,
                                 shuffle=(key == "train"),
                                 num_workers=num_workers,
                                 pin_memory=torch.cuda.is_available())

    # Vị trí metadata hospital trong tensor metadata
    # Camelyon17 metadata: column 0 = hospital
    hospital_idx = 0
    n_groups = 5  # 5 bệnh viện

    return splits, hospital_idx, n_groups
