from torch import Tensor

def get_batch_accuracy(predictions: Tensor, ground_truths: Tensor, batch_size: int):
    """
    Takes predictions and ground truths and computes the accuracy of the predictions in this batch.

    Args:
        predictions (Tensor): Predictions output from the model
        ground_truths (Tensor): Ground truth output from the dataloader
        batch_size (int): size of the current batch

    Returns:
        float: accuracy of the predictions
    """
    pred = predictions.argmax(dim=1)
    # Ensure same dimensions of pred and ground_truths
    correct = (pred == ground_truths.view_as(pred)).sum().item()
    return correct / batch_size