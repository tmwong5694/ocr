import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from src.utils.plot import plot_loss
from tqdm import tqdm

from src.utils.early_stopping import EarlyStopping
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
        save_path: Path = "best_model.pth",
        class_mapping: dict = None,
        early_stopping: EarlyStopping = None,
        plot_save_path: Path = "best_model.png"
):

    model = model.to(device)

    best_val_loss = float('inf')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    logger.info("Starting training on device: {}", device)
    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        logger.info("\nEpoch {}/{}", epoch + 1, num_epochs)
        logger.info("-" * 20)

        model.train()
        epoch_train_loss = 0.0
        correct_train_count = total_train_count = 0

        train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}", leave=False)
        for batch_idx, (inputs, labels) in enumerate(train_loop):
            inputs, labels = inputs.to(device), labels.to(device)
            # Last batch can be equal to or less than predefined batch_size in loader
            current_batch_size = labels.size(0)

            # Reset the parameters gradient o prevent accumulation
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Add the mean of loss * current batch size to the accumulated loss
            epoch_train_loss += loss.item() * current_batch_size
            total_train_count += current_batch_size

            # Add the mean of accuracy * size of batch to the accumulated correct counts
            batch_acc = get_batch_accuracy(outputs, labels, current_batch_size)
            correct_train_count += batch_acc * current_batch_size
            # Update the loss and accuracy at the end of each batch
            train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")
            if (batch_idx + 1) % 50 == 0:
                logger.info("Train Batch {}/{} | Loss: {:.4f}", batch_idx + 1, len(train_loader), loss.item())

        epoch_train_loss = epoch_train_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train_count / total_train_count



        model.eval()
        epoch_val_loss = 0.0
        correct_val_count = total_val_count = 0


        logger.info("  Running validation...")

        with torch.no_grad():

            val_loop = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}", leave=False)
            for (inputs, labels) in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                # Last batch can be equal to or less than predefined batch_size in loader
                current_batch_size = labels.size(0)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                epoch_val_loss += loss.item() * current_batch_size
                total_val_count += current_batch_size

                batch_acc = get_batch_accuracy(outputs, labels, current_batch_size)
                correct_val_count += batch_acc * current_batch_size

        epoch_val_loss = epoch_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val_count / total_val_count



        logger.info("Train Loss: {:.4f} | Train Acc: {:.4f}", epoch_train_loss, epoch_train_acc)
        logger.info("Val Loss:   {:.4f} | Val Acc:   {:.4f}", epoch_val_loss, epoch_val_acc)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        if plot_save_path:
            plot_loss(
                train_loss=history['train_loss'],
                val_loss=history['val_loss'],
                save_path=plot_save_path,
                show=False
            )


        checkpoint = {
            'model_state_dict': model.state_dict(),
            'class_to_idx': class_mapping
        }

        # Save the model if validation loss decreased
        if early_stopping is not None:
            should_stop = early_stopping(val_loss=epoch_val_loss, model=model, save_path=save_path, checkpoint=checkpoint)
            if should_stop:
                break
        else:
            # Fallback if no early stopping was requested
            torch.save(checkpoint, save_path)

    # Calculate elapsed time using perf_counter
    time_elapsed = time.perf_counter() - start_time
    logger.info("\nTraining complete in {:.0f}m {:.0f}s", time_elapsed // 60, time_elapsed % 60)
    logger.info("Best val_loss: {:.4f}", best_val_loss)

    return model, history