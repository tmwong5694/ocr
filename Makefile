# Default configuration path
CONFIG ?= .config/train_config.yaml

.PHONY: help train clean-data

help:
	@echo "Available commands:"
	@echo "  make train                 - Run the training pipeline with the default config"
	@echo "  make clean-data            - Scan and remove corrupted images from the dataset"


train:
	uv run python train.py --config $(CONFIG)

clean-data:
	uv run python clean_data.py