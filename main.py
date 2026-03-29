import torch
import torch.nn as nn
import torch.optim as optim

# Import from your structured src directory
from src.data.dataset import get_dataloaders
from src.models.resnet import TransferResNet
from src.engine.trainer import train_model


def main():
    # ==========================================
    # 1. Configuration & Hyperparameters
    # ==========================================
    DATA_DIR = "./data"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    SAVE_PATH = "best_model.pth"

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
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\nInitializing ResNet18...")
    # Freeze the backbone for transfer learning (only trains the final fc layer)
    model = TransferResNet(num_classes=2, freeze=True)

    # ==========================================
    # 4. Loss Function and Optimizer
    # ==========================================
    criterion = nn.CrossEntropyLoss()

    # Filter out frozen parameters so the optimizer doesn't waste resources tracking them
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

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
        num_epochs=NUM_EPOCHS,
        device=device,
        save_path=SAVE_PATH
    )

    print(f"\nPipeline complete! The optimal weights are saved in '{SAVE_PATH}'.")
    print("Next step: Create inference.py to evaluate the test_loader.")


if __name__ == "__main__":
    main()