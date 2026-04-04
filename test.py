import argparse
import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger

from src.data.dataset import get_dataloaders
from src.models.factory import get_model
from src.utils.config import load_config
from src.utils.logger import set_loguru


def evaluate_test_set(config_path: str | Path):
    cfg = load_config(config_path)

    # 1. Setup paths
    experiment_dir = Path(cfg['paths']['experiments_root']) / cfg['experiment_name'] / cfg['run_name']
    weights_path = experiment_dir / cfg['paths'].get('checkpoints_dirname', 'checkpoints') / "best_model.pth"
    log_path = experiment_dir / cfg['paths'].get('logs_dirname', 'logs') / "test.log"

    set_loguru(level="info", logger_path=log_path)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    # 2. Get Test Dataloader
    logger.info("Loading test dataset...")
    _, _, test_loader = get_dataloaders(
        data_dir=cfg['data']['dataset_dir'],
        batch_size=cfg['data']['batch_size'],
        seed=cfg['seed'],
        num_workers=cfg['data']['num_workers']
    )

    # 3. Model setup
    logger.info("Loading model architecture '{}'...", cfg['model']['name'])
    model = get_model(cfg['model']['name'], cfg['model']['params'])
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 4. Evaluation Loop
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    logger.info("Starting evaluation on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.view_as(predicted)).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    logger.info("Test Set Results | Loss: {:.4f} | Accuracy: {:.4f} ({}/{})", avg_loss, accuracy, correct, total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=".config/train_config.yaml")
    args = parser.parse_args()
    evaluate_test_set(args.config)
