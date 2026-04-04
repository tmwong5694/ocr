import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datetime import date
from pathlib import Path

from src.data.dataset import get_dataloaders
from src.engine.trainer import train_model 
from src.models.transfer_resnet import TransferResNet
from src.models.cat_dog_classifier import CatDogClassifier
from src.utils.plot import plot_loss
from loguru import logger 
from src.utils.set_logger import set_loguru



def get_model(cfg: dict):
    model_name = cfg['model']['name']
    model_name_lower = model_name.lower()
    model_params = cfg['model']['params']
    
    if model_name_lower == "transferresnet":
        return TransferResNet(**model_params)
    elif model_name_lower == "catdogclassifier":
        return CatDogClassifier(**model_params)
    else:
        raise ValueError(f"Model {model_name} is not supported!")


def load_config(config_path) -> dict:
    """Safely loads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main(config_path: Path | str) -> None:
    # 1. Setup Environment
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file '{config_path}' does not exist")
    cfg = load_config(config_path)

    log_dir = (
        Path(cfg['paths']['experiments_root']) /
        cfg['experiment_name'] /
        cfg['run_name'] /
        cfg['paths'].get('logs_dirname', 'logs')
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_{cfg['model']['name']}.log"

    set_loguru(level="info", logger_path=log_path)

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    logger.info("--- PyTorch Image Classification ---\nUsing device: {}", DEVICE)

    # 2. Data Pipeline
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg['data']['dataset_dir'],
        batch_size=cfg['data']['batch_size'],
        seed=cfg['seed'],
        num_workers=cfg['data']['num_workers']
    )
    logger.info("Train: {} | Val: {} | Test: {}", len(train_loader), len(val_loader), len(test_loader))

    # 3. Model Initialization
    logger.info("Initializing {}...", cfg['model']['name'])
    model = get_model(cfg)

    # 4. Training Components
    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=cfg['training']['optimizer']['learning_rate'])

    # 5. Execute Training Engine
    save_dir = (
        Path(cfg['paths']['experiments_root']) /
        cfg['experiment_name'] /
        cfg['run_name'] /
        cfg['paths']['checkpoints_dirname']
    )
    save_path = save_dir / "best_model.pth"

    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting training engine...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg['training']['num_epochs'],
        device=DEVICE,
        save_path=save_path
    )

    image_dir = Path("outputs") / "plots"
    image_name = f"loss_{cfg['model']['name']}_epoch{cfg['training']['num_epochs']}_{date.today():%Y.%m.%d}.jpeg"

    plot_loss(train_loss=history['train_loss'], val_loss=history["val_loss"], save_path=image_dir / image_name)

    logger.info("\nPipeline complete! Weights saved in '{}'.", save_path)


if __name__ == "__main__":
    # Set up argument parsing to accept the config file from the terminal
    parser = argparse.ArgumentParser(description="Train a PyTorch Image Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default=Path(".config") / "train_config.yaml",
        help="Path to the YAML configuration file"
    )

    args = parser.parse_args()
    main(args.config)