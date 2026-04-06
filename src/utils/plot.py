from pathlib import Path
from torchmetrics.classification import BinaryConfusionMatrix, MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_loss(
        train_loss: list,
        val_loss: list,
        figsize: tuple = (8, 6),
        title: str = "Loss vs Epochs",
        xlabel: str = "Epochs",
        ylabel: str = "Loss",
        save_path: str | Path | None = None,
        show: bool = False
) -> None:
    
    plt.figure(figsize=figsize)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    if save_path:
        save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=False)
        plt.savefig(save_path)

    if show:
        # plt.show() clears the canvas, need to put at the end
        plt.show()
    # Close the active canvas
    plt.close()

def get_confusion_matrix(
        num_classes: int,
        normalize: str
):
    num_classes = num_classes
    if num_classes == 2:
        cm = BinaryConfusionMatrix(normalize=normalize)
    elif num_classes > 2:
        cm = MulticlassConfusionMatrix(num_classes, normalize=normalize)
    return cm

def plot_confusion_matrix(
        conv_array: np.ndarray,
        class_names: list[str],
        figsize: tuple = (8, 6),
        title: str = "Confusion Matrix",
        xlabel: str = "True Label",
        ylabel: str = "Predicted Label",
        save_path: str | Path | None = None,
        show: bool = False
) -> None:

    plt.figure(figsize=figsize)
    sns.heatmap(
        conv_array,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close()