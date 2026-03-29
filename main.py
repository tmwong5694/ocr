import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

# Import from your structured src directory
from src.data.dataset import get_dataloaders
from src.models.resnet import TransferResNet
from src.engine.trainer import train_model

def load_config(config_path):
    """Safely loads the YAML configuration file into a dictionary."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main(config_path):
    # ==========================================
    # 1. Configuration & Hyperparameters
    # ==========================================
    cfg = load_config(config_path)

    # Automatically select the best available hardware accelerator
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon (M1/M2/M3)
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # For NVIDIA GPUs
    else:
        device = torch.device("cpu")  # Fallback

    print(f"--- PyTorch Image Classification ---")
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Pipeline
    # ==========================================
    print("\nLoading datasets...")
    # This returns our train, val, and test loaders.
    # The test_loader is safely kept out of the training loop.
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg['data_dir'],
        batch_size=cfg['batch_size'],
        seed=cfg['seed'],
        num_workers=cfg['num_workers']
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\nInitializing ResNet18...")
    # Freeze the backbone for transfer learning (only trains the final fc layer)
    model = TransferResNet(num_classes=cfg['num_classes'], freeze=cfg['freeze'])

    # ==========================================
    # 4. Loss Function and Optimizer
    # ==========================================
    criterion = nn.CrossEntropyLoss()

    # Filter out frozen parameters so the optimizer doesn't waste resources tracking them
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=cfg['learning_rate'])

    # ==========================================
    # 5. Execute Training Engine
    # ==========================================
    print("\nStarting training engine...")
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

    print(f"\nPipeline complete! The optimal weights are saved in '{cfg['save_path']}'.")
    print("Next step: Create inference.py to evaluate the test_loader.")


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