import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import v2

device = torch.device("mps")

transform = transforms.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

# ImageFolder automatically assigns labels based on the folder names (Cat/Dog)
full_dataset = datasets.ImageFolder(root="./data", transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(
    dataset=full_dataset,
    lengths=[train_size, test_size],
    generator=generator
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
)
test_loader = DataLoader(
dataset=test_dataset,
    batch_size=32,
    shuffle=False,
)

pass