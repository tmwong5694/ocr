# Default configuration path
CONFIG ?= ml/.config/train_config.yaml
DETECT_CONFIG ?= ml/.config/yolo_config.yaml
IMAGE ?= ml/samples/husky.jpeg

.PHONY: help train clean-data
.PHONY: api-run api-dev api-test

help:
	@echo "Available commands:"
	@echo "  make train                 - Run the training pipeline with the default config"
	@echo "  make train CONFIG=path.yml - Run the training pipeline with a custom config"
	@echo "  make test				    - Run the testing pipeline with the default config"
	@echo "  make infer IMAGE=path.png  - Run inference on a specific image"
	@echo "  make clean-data            - Scan and remove corrupted images from the dataset"
	@echo "  make label IMAGE=path.png - Run YOLO object detection locally using detect.yaml"
	@echo "  make api-run               - Start the API server"
	@echo "  make api-dev               - Start the API server in development mode with hot reloading"
	@echo "  make api-test              - Run backend API tests"

train:
	uv run ml-train --config $(CONFIG)

test:
	uv run ml-test --config $(CONFIG)

infer:
	uv run ml-infer --config $(CONFIG) --image $(IMAGE)

label:
	uv run ml-label --config $(DETECT_CONFIG)

clean-data:
	uv run ml-clean



api-run:
	cd backend && uv run uvicorn ocr_backend.main:app --host 0.0.0.0 --port 8000

api-dev:
	cd backend && uv run uvicorn ocr_backend.main:app --reload --host 0.0.0.0 --port 8000

api-test:
	cd backend && uv run pytest