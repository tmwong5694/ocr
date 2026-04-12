import argparse
import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger

from ml_pipeline.data.dataset import get_dataloaders
from ml_pipeline.models.factory import get_model
from ml_pipeline.utils.config import load_config
from ml_pipeline.utils.metrics import get_batch_accuracy
from ml_pipeline.utils.plot import get_confusion_matrix, plot_confusion_matrix
from ml_pipeline.utils.logger import set_loguru


def evaluate_test_set(config_path: str | Path) -> None:
    """
    Takes in the config path and evaluate the loss and accuracy on the test set.
    The config contains path to trained model weightings and path to log.
    Test set are parsed into the model and results are logged.

    Args:
        config_path (str | Path): The path to the config file.
    """
    cfg = load_config(config_path)

    # 1. Setup paths
    experiment_dir = Path(cfg['paths']['experiments_root']) / cfg['experiment_name'] / cfg['run_name']
    weights_path = experiment_dir / cfg['paths'].get('checkpoints_dirname', 'checkpoints') / "best_model.pth"
    log_path = experiment_dir / cfg['paths'].get('logs_dirname', 'logs') / "test.log"

    set_loguru(level="info", logger_path=log_path)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    # 2. Get Test Dataloader
    logger.info("Loading test dataset...")
    _, _, test_loader = get_dataloaders(
        dataset_name=cfg["data"]["dataset_name"],
        data_dir=cfg["data"]["dataset_dir"],
        split_ratio = [0.8, 0.1, 0.1],
        batch_size=cfg["data"]["batch_size"],
        seed=cfg["seed"],
    )

    # 3. Model setup
    logger.info("Loading model architecture '{}'...", cfg['model']['name'])
    model = get_model(cfg['model']['name'], cfg['model']['params'])
    pickled_object = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(pickled_object["model_state_dict"])
    model.to(device)
    model.eval()

    class_names = [class_name for class_name in pickled_object['class_to_idx']]

    # 4. Evaluation Loop
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct_count = total_count = 0

    num_classes = len(class_names)
    conv_mtx = get_confusion_matrix(num_classes=num_classes, normalize="true").to(device)

    logger.info("Starting evaluation on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Last batch can be equal to or less than predefined batch_size in loader
            current_batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Add the mean of loss * current batch size to the accumulated loss
            total_loss += loss.item() * current_batch_size
            total_count += current_batch_size

            # Add the mean of accuracy * size of batch to the accumulated correct counts
            batch_accuracy, predictions = get_batch_accuracy(outputs, labels, current_batch_size)
            correct_count += batch_accuracy * current_batch_size
            # Update on the fly to save memory
            conv_mtx.update(predictions, labels)

    conv_mtx_dense = conv_mtx.compute().cpu().numpy()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct_count / total_count

    logger.info("Test Set Results | Loss: {:.4f} | Accuracy: {:.4f} ({}/{})", avg_loss, accuracy, correct_count, total_count)


    cm_save_path = experiment_dir / cfg['paths'].get('artifacts_dirname', 'artifacts') / "test_confusion_matrix.jpeg"
    plot_confusion_matrix(
        conv_array=conv_mtx_dense,
        class_names=class_names,
        save_path=cm_save_path,
        show=False
    )

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml/.config/train_config.yaml")
    args = parser.parse_args()
    evaluate_test_set(args.config)

if __name__ == "__main__":
    cli_main()
