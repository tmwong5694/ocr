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

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    dataset=full_dataset,
    lengths=[0.8, 0.1, 0.1],
    generator=generator
)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)



pass