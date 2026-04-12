import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from ml_pipeline.data.dataset import get_transforms
from ml_pipeline.models.factory import get_model
from ml_pipeline.utils.config import load_config


def infer_image(
        image_path: Path | str,
        weights_path: Path | str,
        device: torch.device,
        model_name: str
) -> tuple[str, float]:
    """
    Infer an image using a trained model.
    Loads the image to infer, weights of a pretrained model, and returns the predicted class and confidence score.

    Args:
        image_path (Path | str): Path to the image to infer.
        weights_path (Path | str): Path to the pretrained weights.
        device (torch.device): Device to use.
        model_name (str): Name of the model.

    Returns:
        tuple[str, float]: Predicted class and confidence score.

    Raises:
        FileNotFoundError: If the image or weights file does not exist.
        RuntimeError: If the image cannot be loaded.
    """
    image_path = Path(image_path)
    weights_path = Path(weights_path)

    if not image_path.is_file():
        raise FileNotFoundError("Image does not exist")
    if not weights_path.is_file():
        raise FileNotFoundError("Weights does not exist")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Reverse class_to_idx to idx_to_class
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model_params = {"num_classes": num_classes}
    if model_name.lower().replace("_", "") == "transferresnet":
        model_params["freeze"] = True

    model = get_model(model_name=model_name, model_params=model_params)

    # Load the pretrained weights
    model.load_state_dict(checkpoint['model_state_dict'])
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
    predicted_class = idx_to_class[predicted_idx.item()]
    confidence_score = confidence.item() * 100.0

    return predicted_class, confidence_score


def cli_main():
    # 1. Setup argument parser so you can pass different images from the terminal
    parser = argparse.ArgumentParser(description="Run inference on an image")
    parser.add_argument(
        "--config",
        type=str,
        default="ml/.config/train_config.yaml",
        help="Path to the config file containing model details"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,  # Make this required since the Makefile provides it
        help="Path to the image relative to the monorepo root"
    )
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.config)

    # Reconstruct the weights path exactly like you did in test.py
    experiment_dir = Path(cfg['paths']['experiments_root']) / cfg['experiment_name'] / cfg['run_name']
    weights_path = experiment_dir / cfg['paths'].get('checkpoints_dirname', 'checkpoints') / "best_model.pth"
    model_name = cfg['model']['name']

    # 2. Determine device
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    # 3. Run inference
    try:
        predicted_class, confidence = infer_image(
            image_path=args.image,
            weights_path=weights_path,
            device=DEVICE,
            model_name=model_name
        )
        print(f"predicted class: {predicted_class}, probability: {confidence:.2f}%")

    except FileNotFoundError as e:
        print(f"Error: {e}. Are you running this from the monorepo root?")


if __name__ == "__main__":
    cli_main()
