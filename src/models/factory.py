from src.models.cat_dog_classifier import CatDogClassifier
from src.models.transfer_resnet import TransferResNet

def get_model(model_name: str, model_params: dict):
    model_name_lower = model_name.lower()

    if model_name_lower == "transferresnet":
        return TransferResNet(**model_params)
    elif model_name_lower == "catdogclassifier":
        return CatDogClassifier(**model_params)
    else:
        raise ValueError(f"Model {model_name} is not supported!")