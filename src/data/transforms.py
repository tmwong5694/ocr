import torch
from torchvision.transforms import v2

def get_transforms(img_size: tuple = (224, 224)):

    train_transform = v2.Compose([
        v2.Resize(img_size),
        v2.RandomHorizontalFlip(p=0.5),  # Safe to add augmentations here later
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    eval_transform = v2.Compose([
        v2.Resize(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    return {"train": train_transform, "eval": eval_transform}