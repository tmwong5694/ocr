import torch
from torchvision.transforms import v2

def get_transforms(img_size: tuple = (224, 224)):

    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = v2.Compose([
        v2.Resize(img_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=20),
        v2.RandomApply([v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize
    ])

    eval_transform = v2.Compose([
        v2.Resize(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize
    ])

    return {"train": train_transform, "eval": eval_transform}