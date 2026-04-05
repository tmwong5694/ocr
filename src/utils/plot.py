from pathlib import Path
import matplotlib.pyplot as plt

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