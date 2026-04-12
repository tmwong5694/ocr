# Default configuration path
CONFIG ?= ml/.config/train_config.yaml

.PHONY: help train clean-data

help:
	@echo "Available commands:"
	@echo "  make train                 - Run the training pipeline with the default config"
	@echo "  make train CONFIG=path.yml - Run the training pipeline with a custom config"
	@echo "  make test				    - Run the testing pipeline with the default config"
	@echo "  make clean-data            - Scan and remove corrupted images from the dataset"


train:
	uv run ml-train --config $(CONFIG)

test:
	uv run ml-test --config $(CONFIG)

clean-data:
	uv run ml-clean