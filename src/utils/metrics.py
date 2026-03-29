def get_batch_accuracy(output, y, N):

    pred = output.argmax(dim=1)
    correct = (pred == y.view_as(pred)).sum().item()
    return correct / N