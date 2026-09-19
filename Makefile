# Default configuration path
CONFIG ?= ml/.config/train_config.yaml
DETECT_CONFIG ?= ml/.config/yolo_config.yaml
OCR_CONFIG ?= ml/.config/detect_config.yaml

.PHONY: help train test classify label clean-data ocr
.PHONY: api-run api-dev api-test frontend-build
.PHONY: docker-up docker-stop docker-down docker-build docker-logs update

help:
	@echo "Available commands:"
	@echo "  make train                    - Run the training pipeline with the default config"
	@echo "  make train CONFIG=path.yml    - Run the training pipeline with a custom config"
	@echo "  make test                     - Run the testing pipeline with the default config"
	@echo "  make classify                 - Run inference on image specified in config"
	@echo "  make classify CONFIG=path.yml - Run inference with custom config"
	@echo "  make clean-data               - Scan and remove corrupted images from the dataset"
	@echo "  make label                    - Run YOLO object detection on image specified in config"
	@echo "  make label CONFIG=path.yml    - Run YOLO detection with custom config"
	@echo "  make ocr                      - Run OCR on image specified in config"
	@echo "  make api-run                  - Start the API server"
	@echo "  make api-dev                  - Start the API server with hot reloading"
	@echo "  make api-test                 - Run backend API tests"
	@echo "  make frontend-build           - Compile the TypeScript frontend"
	@echo "  make docker-up                - Start Docker Compose services"
	@echo "  make docker-stop              - Stop containers without removing them"
	@echo "  make docker-down              - Stop and remove Compose containers"
	@echo "  make docker-build             - Build Compose images using cache"
	@echo "  make docker-logs              - Follow Compose service logs"
	@echo "  make update                   - Pull, rebuild if needed, and restart services"

# ML pipeline
train:
	uv run ml-train --config $(CONFIG)

test:
	uv run ml-test --config $(CONFIG)

classify:
	uv run ml-classify --config $(CONFIG)

label:
	uv run ml-label --config $(DETECT_CONFIG)

clean-data:
	uv run ml-clean

ocr:
	uv run ml-ocr --config $(OCR_CONFIG)

# Backend
api-run:
	uv run --package backend uvicorn ocr_backend.main:app --host 0.0.0.0 --port 8000

api-dev:
	cd backend && uv run uvicorn ocr_backend.main:app --reload --host 0.0.0.0 --port 8000

api-test:
	cd backend && uv run pytest

# Frontend
frontend-build:
	cd frontend && npm install && npm run build

# Docker
docker-up: frontend-build
	docker compose up -d --build

docker-stop:
	docker compose stop

docker-down:
	docker compose down

docker-build: frontend-build
	docker compose build

docker-logs:
	docker compose logs -f

update:
	git pull
	$(MAKE) docker-up