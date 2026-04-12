from ml_pipeline.models.cat_dog_classifier import CatDogClassifier
from ml_pipeline.models.transfer_resnet import TransferResNet
from ml_pipeline.models.mnist_classifier import MNISTClassifier


MODEL_DISPATCH = {
    "transferresnet": TransferResNet,
    "catdogclassifier": CatDogClassifier,
    "mnistclassifier": MNISTClassifier,
}


def get_model(model_name: str, model_params: dict):
    model_name_clean = model_name.lower().replace("_", "")

    model_chose = MODEL_DISPATCH.get(model_name_clean)

    if model_chose is None:
        raise ValueError(f"Model {model_name} is not supported!")

    return model_chose(**model_params)