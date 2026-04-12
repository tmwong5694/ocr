from ml_pipeline.models.cat_dog_classifier import CatDogClassifier
from ml_pipeline.models.transfer_resnet import TransferResNet

def get_model(model_name: str, model_params: dict):
    model_name_clean = model_name.lower().replace("_", "")

    if model_name_clean == "transferresnet":
        return TransferResNet(**model_params)
    elif model_name_clean == "catdogclassifier":
        return CatDogClassifier(**model_params)
    else:
        raise ValueError(f"Model {model_name} is not supported!")