from torch import Tensor

def get_batch_accuracy(y_pred: Tensor, y_label: Tensor, batch_size: int) -> tuple[float, Tensor]:
    """
    Takes predictions and ground truths and computes the accuracy of the predictions in this batch.

    Args:
        y_pred (Tensor): Predictions output from the model
        y_label (Tensor): Ground truth output from the dataloader
        batch_size (int): size of the current batch

    Returns:
        float: accuracy of the predictions
        Tensor: tensor of predicted class (shape: (n, 1)
    """
    pred = y_pred.argmax(dim=1)
    # Ensure same dimensions of pred and ground_truths
    correct = (pred == y_label.view_as(pred)).sum().item()
    batch_acc = correct / batch_size
    return batch_acc, pred