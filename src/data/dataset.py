import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from src.data.transforms import get_transforms

def get_dataloaders(data_dir="./data", batch_size=64, split_ratios=[0.8, 0.1, 0.1], seed=42, num_workers=None):
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


    # # Set up num_workers automatically if not provided
    # if num_workers is None:
    #     # Default to 4 or the number of CPU cores available (whichever is lower)
    #     num_workers = min(4, os.cpu_count() or 1)
    #     print(f"Auto-configured DataLoaders to use {num_workers} worker processes.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader