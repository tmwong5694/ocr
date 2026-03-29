import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import get_dataloaders
from src.models.resnet import TransferResNet
from src.engine.trainer import train_model


def load_config(config_path):
    """Safely loads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main(config_path):
    # 1. Setup Environment
    cfg = load_config(config_path)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"--- PyTorch Image Classification ---\nUsing device: {device}")

    # 2. Data Pipeline
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg['data_dir'],
        batch_size=cfg['batch_size'],
        seed=cfg['seed'],
        num_workers=cfg['num_workers']
    )
    print(f"Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # 3. Model Initialization
    print("\nInitializing ResNet18...")
    model = TransferResNet(num_classes=cfg['num_classes'], freeze=cfg['freeze'])

    # 4. Training Components
    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=cfg['learning_rate'])

    # 5. Execute Training Engine
    print("Starting training engine...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg['num_epochs'],
        device=device,
        save_path=cfg['save_path']
    )

    print(f"\nPipeline complete! Weights saved in '{cfg['save_path']}'.")


if __name__ == "__main__":
    # Set up argument parsing to accept the config file from the terminal
    parser = argparse.ArgumentParser(description="Train a PyTorch Image Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default=".config/train_config.yaml",
        help="Path to the YAML configuration file"
    )

    args = parser.parse_args()
    main(args.config)