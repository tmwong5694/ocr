import torch

from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets
from src.data.transforms import get_transforms

def _get_imagefolder_datasets(
        data_dir: str | Path,
        train_transform,
        eval_transform,
        split_ratio: list[int | float] | tuple[int | float, ...],
        seed: int = 42
) -> tuple[Dataset, Dataset, Dataset]:

    # Instantiate all at once with respective transformation method
    train_data = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_data = datasets.ImageFolder(root=data_dir, transform=eval_transform)
    test_data = datasets.ImageFolder(root=data_dir, transform=eval_transform)

    # Use indices to select respective train, val or test data
    total_indices = range(len(train_data))
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices, test_indices = random_split(
        dataset=total_indices,
        lengths=split_ratio,
        generator=generator
    )

    train_dataset = Subset(train_data, indices=train_indices)
    val_dataset = Subset(val_data, indices=val_indices)
    test_dataset = Subset(test_data, indices=test_indices)

    return train_dataset, val_dataset, test_dataset

def _get_mnist_datasets(
        data_dir: str | Path,
        train_transform,
        eval_transform,
        split_ratio: list[int | float] | tuple[int | float, ...],
        seed: int = 42
) -> tuple[Dataset, Dataset, Dataset]:
    
    to_download = True
    train_dataset_full = datasets.MNIST(data_dir, train=True, download=to_download, transform=train_transform)
    val_dataset_full = datasets.MNIST(data_dir, train=True, download=to_download, transform=eval_transform)

    split_indices = range(len(train_dataset_full))
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        dataset=split_indices,
        lengths=split_ratio,
        generator=generator
    )

    train_data = Subset(train_dataset_full, train_indices)
    val_data = Subset(val_dataset_full, val_indices)
    train_dataset = Subset(train_data, indices=train_indices)
    val_dataset = Subset(val_data, indices=val_indices)

    test_dataset = datasets.MNIST(data_dir, train=False, download=to_download, transform=eval_transform)
    
    return train_dataset, val_dataset, test_dataset

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
        dataset_name: str,
        data_dir: str | Path = "./data",
        split_ratio: list[int | float] | tuple[int | float, ...] = None,
        batch_size: int = 64,
        seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:

    if split_ratio is None:
        raise ValueError("Split ratio cannot be None")

    transforms_dict = get_transforms()
    train_transform = transforms_dict['train']
    eval_transform = transforms_dict['eval']

    if dataset_name.lower() in ("imagefolder", "image_folder"):
        train_dataset, val_dataset, test_dataset = _get_imagefolder_datasets(
            data_dir=data_dir,
            train_transform=train_transform,
            eval_transform=eval_transform,
            split_ratio=split_ratio,
            seed=seed
        )
    elif dataset_name.lower() == "mnist":
        train_dataset, val_dataset, test_dataset = _get_mnist_datasets(
            data_dir=data_dir,
            train_transform=train_transform,
            eval_transform=eval_transform,
            split_ratio=split_ratio,
            seed=seed
        )

    train_loader, val_loader, test_loader = _get_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
    )

    return train_loader, val_loader, test_loader