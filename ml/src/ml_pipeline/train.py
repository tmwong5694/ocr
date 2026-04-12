import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loguru import logger
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ml_pipeline.data.dataset import get_dataloaders
from ml_pipeline.engine.trainer import train_model
from ml_pipeline.models.factory import get_model
from ml_pipeline.utils.config import load_config
from ml_pipeline.utils.early_stopping import EarlyStopping
from ml_pipeline.utils.logger import set_loguru



def main(config_path: Path | str) -> None:
    # 1. Setup Environment
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file '{config_path}' does not exist")
    cfg = load_config(config_path)

    log_dir = (
        Path(cfg['paths']['experiments_root']) /
        cfg['experiment_name'] /
        cfg['run_name'] /
        cfg['paths'].get('logs_dirname', 'logs')
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_{cfg['model']['name']}.log"

    set_loguru(level="info", logger_path=log_path)

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    logger.info("--- PyTorch Image Classification ---\nUsing device: {}", DEVICE)

    config_yaml_str = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
    logger.info("Configuration settings for this run:\n{}", config_yaml_str)

    # 2. Data Pipeline
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=cfg['data']['dataset_name'],
        data_dir=cfg['data']['dataset_dir'],
        split_ratio = [0.8, 0.1, 0.1],
        batch_size=cfg['data']['batch_size'],
        seed=cfg['seed'],
    )

    # Extract class_to_idx mapping
    class_mapping = train_loader.dataset.dataset.class_to_idx

    logger.info("Train: {} | Val: {} | Test: {}", len(train_loader), len(val_loader), len(test_loader))

    # 3. Model Initialization
    logger.info("Initializing {}...", cfg['model']['name'])
    model = get_model(
        model_name=cfg['model']['name'],
        model_params=cfg['model']['params']
    )

    # 4. Training Components
    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=cfg['training']['optimizer']['learning_rate'])

    # Add scheduler initialization
    scheduler = None
    if 'scheduler' in cfg['training']:
        sch_cfg = cfg['training']['scheduler']
        if sch_cfg['name'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=sch_cfg.get('factor', 0.1),
                patience=sch_cfg.get('scheduler_patience', 10),
                min_lr=float(sch_cfg.get('min_lr', 1e-6))
            )


    # 5. Execute Training Engine
    parent_path = Path(cfg['paths']['experiments_root']) / cfg['experiment_name'] / cfg['run_name']

    save_dir = parent_path / cfg['paths']['checkpoints_dirname']
    save_path = save_dir / "best_model.pth"
    save_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = parent_path / cfg['paths'].get('artifacts_dirname', 'artifacts')
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    loss_curve_path = artifacts_dir / "loss_curve.jpeg"

    early_stopping_patience = cfg['training'].get('early_stop_patience', 10)
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    logger.info("Starting training engine...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg['training']['num_epochs'],
        device=DEVICE,
        save_path=save_path,
        class_mapping=class_mapping,
        early_stopping=early_stopping,
        plot_save_path=loss_curve_path,
        scheduler=scheduler
    )


    logger.info("\nPipeline complete! Weights saved in '{}'.", save_path)


def cli_main():
    parser = argparse.ArgumentParser(description="Train a PyTorch Image Classifier")
    # Because you will run this from the project root using `uv run`, update the path:
    parser.add_argument("--config", type=str, default="ml/.config/train_config.yaml")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli_main()