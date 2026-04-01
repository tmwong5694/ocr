import matplotlib.pyplot as plt

def plot_loss(
        train_loss: list,
        val_loss: list,
        figsize: tuple = (8, 6),
        title: str = "Loss vs Epochs",
        xlabel: str = "Epochs",
        ylabel: str = "Loss"
) -> None:
    
    plt.figure(figsize=figsize)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show() 