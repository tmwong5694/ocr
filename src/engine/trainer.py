import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import get_batch_accuracy
from loguru import logger


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        device: torch.device,
        save_path: Path = "best_model.pth"
):

    model = model.to(device)

    best_val_loss = float('inf')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    logger.info(f"Starting training on device: {device}")
    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 20)

        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}", leave=False)
        for batch_idx, (inputs, labels) in enumerate(train_loop):
            inputs, labels = inputs.to(device), labels.to(device)
            num_samples = labels.size(0)

            # Reset the parameters gradient o prevent accumulation
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_train_loss += loss.item() * num_samples
            total_train += num_samples

            batch_acc = get_batch_accuracy(outputs, labels, num_samples)
            correct_train += batch_acc * num_samples

            train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Train Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train



        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        logger.info("  Running validation...")

        with torch.no_grad():

            val_loop = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}", leave=False)
            for batch_idx, (inputs, labels) in enumerate(val_loop):
                inputs, labels = inputs.to(device), labels.to(device)
                num_samples = labels.size(0)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * num_samples
                total_val += num_samples

                # Robust, Pythonic accuracy calculation
                batch_acc = get_batch_accuracy(outputs, labels, num_samples)
                correct_val += batch_acc * num_samples

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val



        logger.info(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        logger.info(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # Save the model if validation loss decreased
        if epoch_val_loss < best_val_loss:
            logger.info(f"*** Validation loss decreased ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Saving model... ***")
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)

    # Calculate elapsed time using perf_counter
    time_elapsed = time.perf_counter() - start_time
    logger.info(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logger.info(f"Best val_loss: {best_val_loss:.4f}")

    return model, history