from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets
from torchvision.transforms.v2 import Transform
from ml_pipeline.data.transforms import get_transforms_by_dataset

def _get_imagefolder_datasets(
        data_dir: str | Path,
        train_transform: Transform,
        eval_transform: Transform,
        split_ratio: list[int | float],
        seed: int = 42
) -> tuple[Dataset, Dataset, Dataset]:
    """
    For image folder, the data is expected to undergo a three-way train-val-test split.

    Args:
        data_dir (str | Path): Directory with images
        train_transform (torchvision.transforms.v2.Transform): Image transformation
        eval_transform (torchvision.transforms.v2.Transform): Image transformation
        split_ratio (list[int | float]): Ratio of images to use for training, validation and testing. Must be of length 3.
        seed (int): Random seed. Defaults to 42.

    Returns:
        tuple[Dataset, Dataset, Dataset]: Split train dataset, val dataset, test dataset
    """
    # Instantiate all at once with respective transformation method
    train_data = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_data = datasets.ImageFolder(root=data_dir, transform=eval_transform)
    test_data = datasets.ImageFolder(root=data_dir, transform=eval_transform)

    # Class labels and indices
    targets = train_data.targets
    indices = range(len(targets))

    # Calculate the split ratio in float, not whole numbers
    ratio_sum = sum(split_ratio)
    val_pct = split_ratio[1] / ratio_sum
    test_pct = split_ratio[2] / ratio_sum

    # Split out the test_idx first
    # train_val_idx contains indices of train part + val part
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_pct,
        stratify=targets,
        random_state=seed
    )
    # Get the class labels and parse into train_test_split for splitting
    train_val_targets = [targets[i] for i in train_val_idx]
    # Get the proportion of val of the remaining indices of (train + val)
    val_relative_pct = val_pct / (1.0 - test_pct)

    # Split the train and val indices
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative_pct,
        stratify=train_val_targets,
        random_state=seed
    )

    train_dataset = Subset(train_data, indices=train_idx)
    val_dataset = Subset(val_data, indices=val_idx)
    test_dataset = Subset(test_data, indices=test_idx)

    return train_dataset, val_dataset, test_dataset

def _get_mnist_datasets(
        data_dir: str | Path,
        train_transform,
        eval_transform,
        split_ratio: list[int | float],
        seed: int = 42
) -> tuple[Dataset, Dataset, Dataset]:
    """
    The MNIST are presplit with training and testing dataset.
    The function requires a two-way split of the training dataset into train and validation dataset.

    Args:
        data_dir (str | Path): Directory with images
        train_transform (torchvision.transforms.Compose): Image transformation
        eval_transform (torchvision.transforms.Compose): Image transformation
        split_ratio (list[int | float]): Ratio of images to use for training, validation (and testing). Expect the length of 2
        seed (int): Random seed. Defaults to 42.

    Returns:
        tuple[Dataset, Dataset, Dataset]: Split train dataset, val dataset, test dataset
    """
    to_download = True
    train_dataset_full = datasets.MNIST(data_dir, train=True, download=to_download, transform=train_transform)
    val_dataset_full = datasets.MNIST(data_dir, train=True, download=to_download, transform=eval_transform)

    # Class labels and indices
    targets = train_dataset_full.targets
    indices = range(len(targets))

    # Convert list of integers into list of floats
    # Only take first two elements in case 3 ratios are parsed in
    ratio_sum = sum(split_ratio[:2])
    test_pct = split_ratio[1] / ratio_sum

    # Split the train indices and validation indices
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_pct,
        stratify=targets,
        random_state=seed
    )

    train_dataset = Subset(train_dataset_full, indices=train_idx)
    val_dataset = Subset(val_dataset_full, indices=val_idx)
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

    transforms_dict = get_transforms_by_dataset(dataset_name)
    train_transform = transforms_dict['train']
    eval_transform = transforms_dict['eval']

    DATASET_DISPATCH = {
        'mnist': _get_mnist_datasets,
        "imagefolder": _get_imagefolder_datasets
    }

    dataset_name_lower = dataset_name.lower()
    dispatched_func = DATASET_DISPATCH.get(dataset_name_lower)

    if dispatched_func is None:
        valid_keys = list(DATASET_DISPATCH.keys())
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Available options: {valid_keys}")

    train_dataset, val_dataset, test_dataset = dispatched_func(
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