"""
CLI script for vision-language OCR text detection and recognition.

This script provides a command-line interface for running OCR inference
on images using flexible vision-language models (GLM-OCR, Qwen-VL, etc.).

Configuration is loaded from detect_config.yaml in ml/.config/
"""

import argparse
from pathlib import Path
from loguru import logger

from ml_pipeline.ocr_model import OCRModel, DEFAULT_IMAGE_PATH
from ml_pipeline.utils.config import load_config


def cli_main():
    """
    Main entry point for vision-language OCR detection.
    
    Loads configuration from detect_config.yaml and runs text recognition
    on the specified image using the configured model.
    """
    parser = argparse.ArgumentParser(
        description="Run vision-language OCR on images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ml/.config/detect_config.yaml",
        help="Path to the config file containing OCR model settings"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image (overrides config setting)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (overrides config setting)"
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        raise

    # Get image path from argument or config
    image_path = args.image or config.get("ocr_model", {}).get("default_image_path", DEFAULT_IMAGE_PATH)
    model_name = args.model or config.get("ocr_model", {}).get("model_name")
    
    if not image_path:
        raise ValueError("Image path not specified in config or command line")
    
    logger.info(f"Starting vision-language OCR inference")
    logger.info(f"Image: {image_path}")
    logger.info(f"Model: {model_name}")

    # Initialize and run OCR model
    try:
        ocr_model = OCRModel(model_name=model_name)
        ocr_model.load_model()

        result = ocr_model.recognize_text(image_path)

        logger.info(f"Recognition result: {result}")
        print(f"\n{'='*60}")
        print(f"Recognized Text:")
        print(f"{'='*60}")
        print(result)
        print(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Error during OCR inference: {e}")
        raise


if __name__ == "__main__":
    cli_main()
