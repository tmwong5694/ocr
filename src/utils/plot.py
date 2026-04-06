from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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

def plot_confusion_matrix(
        y_true: list,
        y_pred: list,
        class_names: list[str],
        figsize: tuple = (8, 6),
        title: str = "Confusion Matrix",
        xlabel: str = "True Label",
        ylabel: str = "Predicted Label",
        save_path: str | Path | None = None,
        show: bool = False
) -> None:
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize="true")
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
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