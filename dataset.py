"""
Dataset loading and preprocessing for CIFAR-10
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_transforms(augment=False):
    """
    Get data transforms

    Args:
        augment: Whether to apply data augmentation

    Returns:
        transforms.Compose: Composed transforms
    """
    if augment:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])


def get_dataloaders(batch_size=128, val_split=0.1, num_workers=2):
    """
    Get CIFAR-10 data loaders

    Args:
        batch_size: Batch size for training
        val_split: Validation split ratio
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=get_transforms(augment=True)
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=get_transforms(augment=False)
    )

    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
