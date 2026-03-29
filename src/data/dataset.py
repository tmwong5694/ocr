import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from src.data.transforms import get_transforms

transforms_dict = get_transforms()
train_transform = transforms_dict['train']
eval_transform = transforms_dict['eval']

# Instantiate all at once with respective transformation method
train_data = datasets.ImageFolder(root="./data", transform=train_transform)
val_data = datasets.ImageFolder(root="./data", transform=eval_transform)
test_data = datasets.ImageFolder(root="./data", transform=eval_transform)

# Use indices to select respective train, val or test data
total_indices = range(len(train_data))
generator = torch.Generator().manual_seed(42)
train_indices, val_indices, test_indices = random_split(
    dataset=total_indices,
    lengths=[0.8, 0.1, 0.1],
    generator=generator
)

train_dataset = Subset(train_data, indices=train_indices)
val_dataset = Subset(val_data, indices=val_indices)
test_dataset = Subset(test_data, indices=test_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


