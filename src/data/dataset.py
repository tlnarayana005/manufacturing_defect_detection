from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def build_transforms(image_size: int, phase: str) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if phase == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15, fill=0),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.18,
                            contrast=0.18,
                            saturation=0.18,
                            hue=0.05,
                        )
                    ],
                    p=0.75,
                ),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_image_folder(root: Path, split: str, image_size: int, class_to_idx: Optional[dict] = None) -> ImageFolder:
    dataset_root = root / split
    if not dataset_root.exists():
        raise FileNotFoundError(f"Processed split not found: {dataset_root}")
    transform = build_transforms(image_size, split)
    if class_to_idx is not None:
        return ImageFolder(root=dataset_root, transform=transform, class_to_idx=class_to_idx)
    return ImageFolder(root=dataset_root, transform=transform)


def create_weighted_sampler(dataset: ImageFolder) -> WeightedRandomSampler:
    target_counts = Counter(dataset.targets)
    weights = [1.0 / target_counts[target] for target in dataset.targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders(
    root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 4,
    balance_classes: bool = False,
):
    train_dataset = load_image_folder(root, "train", image_size)
    class_to_idx = train_dataset.class_to_idx
    val_dataset = load_image_folder(root, "val", image_size, class_to_idx=class_to_idx)
    test_dataset = load_image_folder(root, "test", image_size, class_to_idx=class_to_idx)

    train_sampler = create_weighted_sampler(train_dataset) if balance_classes else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def get_class_labels(dataset: ImageFolder) -> Tuple[list[str], dict[int, str]]:
    labels = dataset.classes
    reverse_map = {index: label for index, label in enumerate(labels)}
    return labels, reverse_map
