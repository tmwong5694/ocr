import matplotlib.pyplot as plt

def plot_loss(
        train_loss: list,
        val_loss: list,
        figsize: tuple = (8, 6),
        title: str = "Loss vs Epochs",
        xlabel: str = "Epochs",
        ylabel: str = "Loss"
) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show() 