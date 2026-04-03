import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from src.data.transforms import get_transforms
from src.models.transfer_resnet import TransferResNet


DEFAULT_CLASSES = ['Cat', 'Dog']


def infer_image(
        image_path: Path | str,
        weights_path: Path | str,
        device: torch.device,
        class_names: list[str]
):
    image_path = Path(image_path)
    weights_path = Path(weights_path)

    if not image_path.is_file():
        raise FileNotFoundError("Image does not exist")
    if not weights_path.is_file():
        raise FileNotFoundError("Weights does not exist")

    model = TransferResNet(num_classes=len(class_names), freeze=True)

    # # Load the pretrained weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Get eval transformation
    eval_transform = get_transforms()['eval']

    try:
        # Convert to RGB to guarantee 3 channels
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image at '{image_path}' does not exist")
    except Exception as e:
        raise RuntimeError(f"Failed to open image '{image_path}': {e}")

    # Transform and add batch dimension: [C, H, W] -> [1, C, H, W]
    input_tensor = eval_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        # Softmax for multiclass classification
        probabilities = F.softmax(outputs[0], dim=0)
        # Max probability, index of max probability
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    # Get the class prediction and convert score to %
    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item() * 100.0

    return predicted_class, confidence_score



if __name__ == "__main__":
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    sample_folder = Path("samples")
    weight = Path("experiments") / "run_01" / "best_model.pth"

    predicted_class, confidence = infer_image(
        image_path=sample_folder / "mofusand.png",
        weights_path=weight,
        device=DEVICE,
        class_names=DEFAULT_CLASSES
    )

    print(f"predicted class: {predicted_class}, probability: {confidence}")