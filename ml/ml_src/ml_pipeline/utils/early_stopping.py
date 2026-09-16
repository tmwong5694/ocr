from loguru import logger
import torch
from pathlib import Path


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience (int): Number of epochs to wait before stopping if validation loss doesn't improve.
            min_delta (float): Minimum change in the watched quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module, save_path: Path, checkpoint: dict = None) -> bool:
        """
        Returns True if training should stop.
        """
        # First epoch or a significant improvement
        if val_loss < self.best_loss - self.min_delta:
            logger.info(
                "*** Validation loss decreased ({:.4f} --> {:.4f}). Saving model... ***",
                self.best_loss,
                val_loss
            )
            self.best_loss = val_loss
            self.counter = 0
            if checkpoint is not None:
                torch.save(checkpoint, save_path)
            else:
                torch.save(model.state_dict(), save_path)
        else:
            self.counter += 1
            logger.warning("EarlyStopping counter: {} out of {}", self.counter, self.patience)

            if self.counter >= self.patience:
                logger.error("Early stopping triggered. Validation loss has not improved for {} epochs.", self.patience)
                self.early_stop = True

        return self.early_stop
