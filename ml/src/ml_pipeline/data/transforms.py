import torch
from torchvision.transforms import v2

def get_transforms(img_size: int = 224) -> dict[v2.Transform, v2.Transform]:

    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = v2.Compose([
        v2.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0), antialias=True), # Remove jagged lines
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=20),
        v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
        v2.RandomErasing(p=0.2, scale=(0.02, 0.15)) # Must follow ToDtype and normalization
    ])

    eval_transform = v2.Compose([
        # Resize the shortest edge to slightly larger than the target (standard is 256 for 224)
        v2.Resize(int(img_size / 0.875), antialias=True),
        # Crop the exact center to ensure perfectly square, uniform batch sizes
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize
    ])

    return {"train": train_transform, "eval": eval_transform}