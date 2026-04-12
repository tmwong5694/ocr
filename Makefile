# Default configuration path
CONFIG ?= ml/.config/train_config.yaml
IMAGE ?= ml/samples/corgi.png

.PHONY: help train clean-data

help:
	@echo "Available commands:"
	@echo "  make train                 - Run the training pipeline with the default config"
	@echo "  make train CONFIG=path.yml - Run the training pipeline with a custom config"
	@echo "  make test				    - Run the testing pipeline with the default config"
	@echo "  make infer IMAGE=path.png  - Run inference on a specific image"
	@echo "  make clean-data            - Scan and remove corrupted images from the dataset"


train:
	uv run ml-train --config $(CONFIG)

test:
	uv run ml-test --config $(CONFIG)

infer:
	uv run ml-infer --config $(CONFIG) --image $(IMAGE)

clean-data:
	uv run ml-clean