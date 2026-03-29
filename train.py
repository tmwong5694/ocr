import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import v2

device = torch.device("mps")

train_transform = transforms.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(p=0.5), # Safe to add augmentations here later
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

eval_transform = transforms.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

# Instantiate datasets with their respective transforms
# ImageFolder automatically assigns labels based on the folder names (Cat/Dog)
train_data = datasets.ImageFolder(root="./data", transform=train_transform)
val_data = datasets.ImageFolder(root="./data", transform=eval_transform)
test_data = datasets.ImageFolder(root="./data", transform=eval_transform)

generator = torch.Generator().manual_seed(42)
num_samples = len(train_data)

train_indices, val_indices, test_indices = random_split(
    dataset=range(num_samples),
    lengths=[0.8, 0.1, 0.1],
    generator=generator
)

train_dataset = Subset(train_data, train_indices)
val_dataset = Subset(val_data, val_indices)
test_dataset = Subset(test_data, test_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)



pass