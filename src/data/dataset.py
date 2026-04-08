import torch

from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets
from src.data.transforms import get_transforms


def _get_loaders(
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset,
        batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    A helper function to create training, validation and test dataloaders

    Args:
        train_ds (Dataset): the training dataset
        val_ds (Dataset): the validation dataset
        test_ds (Dataset): the test dataset
        batch_size (int): the batch size. Defaults to 64.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]:
        - the training dataloader
        - the validation dataloader
        - the test dataloader
    """

    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False
        ),
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False
        )
    )

def get_dataloaders(
        data_dir: str| Path = "./data",
        batch_size: int = 64,
        split_ratios: list[int | float] | tuple[int | float, ...] = (0.8, 0.1, 0.1),
        seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transforms_dict = get_transforms()
    train_transform = transforms_dict['train']
    eval_transform = transforms_dict['eval']

    # Instantiate all at once with respective transformation method
    train_data = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_data = datasets.ImageFolder(root=data_dir, transform=eval_transform)
    test_data = datasets.ImageFolder(root=data_dir, transform=eval_transform)

    # Use indices to select respective train, val or test data
    total_indices = range(len(train_data))
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices, test_indices = random_split(
        dataset=total_indices,
        lengths=split_ratios,
        generator=generator
    )

    train_dataset = Subset(train_data, indices=train_indices)
    val_dataset = Subset(val_data, indices=val_indices)
    test_dataset = Subset(test_data, indices=test_indices)

    train_loader, val_loader, test_loader = _get_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader

def get_MNIST_loaders(
        data_dir: str | Path = "./data",
        download: bool = True,
        batch_size: int = 64,
        split_ratios: list[int | float] | tuple[int | float, ...] = (50000, 10000),
        seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:

    transforms_dict = get_transforms()
    train_transform = transforms_dict['train']
    eval_transform = transforms_dict['eval']

    train_dataset_full = datasets.MNIST(data_dir, train=True, download=download, transform=train_transform)
    val_dataset_full = datasets.MNIST(data_dir, train=True, download=download, transform=eval_transform)

    split_indices = range(len(train_dataset_full))
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        dataset=split_indices,
        lengths=split_ratios,
        generator=generator
    )

    train_data = Subset(train_dataset_full, train_indices)
    val_data = Subset(val_dataset_full, val_indices)
    train_dataset = Subset(train_data, indices=train_indices)
    val_dataset = Subset(val_data, indices=val_indices)

    test_dataset = datasets.MNIST(data_dir, train=False, download=download, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader