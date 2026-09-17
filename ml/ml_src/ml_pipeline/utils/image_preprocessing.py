"""
Image preprocessing utilities for reducing memory usage in OCR inference.

Provides functions to resize, compress, and optimize images for
vision-language models while maintaining text readability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import json
import re

import numpy as np
from loguru import logger
from PIL import Image


ImageFormat = Literal["JPEG", "PNG", "WEBP"]


def parse_ocr_json(raw: str):
    cleaned = raw.strip()
    cleaned = re.sub(r"^```json|```$", "", cleaned, flags=re.IGNORECASE)
    return json.loads(cleaned)

def get_image_dimensions(image_path: str | Path) -> tuple[int, int]:
    """
    Get the dimensions of an image without decoding the full pixel array.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height)
    """
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        return img.size


def _convert_for_output(img: Image.Image, output_format: ImageFormat) -> Image.Image:
    """
    Convert image mode as needed for the requested output format.

    JPEG does not support alpha channels, so RGBA/LA/P images must be
    converted before saving. For images with transparency, this composites
    onto a white background to preserve text readability on transparent
    regions.
    """
    if output_format in {"PNG", "WEBP"}:
        return img

    # JPEG path: ensure RGB-compatible output
    if img.mode == "RGB":
        return img

    if img.mode in {"RGBA", "LA"}:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background

    if img.mode == "P":
        # Palette images may or may not carry transparency.
        if "transparency" in img.info:
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        return img.convert("RGB")

    return img.convert("RGB")


def resize_image_for_ocr(
    image_path: str | Path,
    output_path: str | Path | None = None,
    max_width: int = 1024,
    max_height: int = 1024,
    quality: int = 85,
    output_format: ImageFormat = "JPEG",
) -> str:
    """
    Resize and optimize an image for OCR to reduce memory usage.

    This function preserves aspect ratio and avoids enlarging smaller images.
    For JPEG output, unsupported modes such as RGBA and some palette images
    are converted to RGB before saving.

    Args:
        image_path: Path to the input image
        output_path: Output file path; if None, a sibling file is created
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels
        quality: Compression quality for JPEG/WEBP, 0-100
        output_format: Output format, one of "JPEG", "PNG", or "WEBP"

    Returns:
        Path to the processed image

    Raises:
        FileNotFoundError: If the image does not exist
        ValueError: If max dimensions are invalid
        PIL.UnidentifiedImageError: If image format is unsupported
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if max_width <= 0 or max_height <= 0:
        raise ValueError("max_width and max_height must be positive integers")

    output_format = output_format.upper()
    if output_format not in {"JPEG", "PNG", "WEBP"}:
        raise ValueError("output_format must be one of: JPEG, PNG, WEBP")

    suffix_map = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}

    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            logger.debug(f"Original image size: {original_width}x{original_height}")

            # Preserve aspect ratio and avoid upscaling smaller images.
            working = img.copy()
            working.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            new_width, new_height = working.size
            if (new_width, new_height) != (original_width, original_height):
                logger.info(
                    f"Resized image from {original_width}x{original_height} "
                    f"to {new_width}x{new_height}"
                )

            working = _convert_for_output(working, output_format)

            if output_path is None:
                output_path = image_path.parent / f"{image_path.stem}_preprocessed{suffix_map[output_format]}"

            output_path = Path(output_path)

            save_kwargs = {"optimize": True}
            if output_format in {"JPEG", "WEBP"}:
                save_kwargs["quality"] = quality

            if output_format == "JPEG":
                save_kwargs["progressive"] = True

            working.save(output_path, format=output_format, **save_kwargs)
            logger.debug(f"Saved preprocessed image to: {output_path}")

            return str(output_path)

    except Exception as e:
        logger.error(f"Failed to preprocess image {image_path}: {e}")
        raise


def estimate_attention_memory_usage(
    image_width: int,
    image_height: int,
    patch_size: int = 14,
    extra_tokens: int = 512,
    bytes_per_element: int = 4,
    overhead_factor: float = 3.0,
) -> float:
    """
    Approximate memory usage for quadratic attention buffers in a vision transformer.

    This is a rough estimate for attention-related tensor storage, not a full-model
    peak-memory predictor. Actual memory use also depends on model architecture,
    hidden size, number of heads, dtype, batch size, framework kernels, and whether
    memory-efficient attention is used.

    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        patch_size: Vision patch size in pixels
        extra_tokens: Non-image tokens added to the sequence
        bytes_per_element: Bytes per tensor element, e.g. 4 for float32
        overhead_factor: Multiplier to approximate related intermediate buffers

    Returns:
        Estimated memory in GB
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if extra_tokens < 0:
        raise ValueError("extra_tokens cannot be negative")
    if bytes_per_element <= 0 or overhead_factor <= 0:
        raise ValueError("bytes_per_element and overhead_factor must be positive")

    num_patches = (image_width // patch_size) * (image_height // patch_size)
    seq_length = num_patches + extra_tokens

    attention_bytes = seq_length * seq_length * bytes_per_element
    total_bytes = attention_bytes * overhead_factor

    return total_bytes / (1024 ** 3)


def suggest_optimal_size(
    target_memory_gb: float = 4.0,
    patch_size: int = 14,
    extra_tokens: int = 512,
    bytes_per_element: int = 4,
    overhead_factor: float = 3.0,
) -> tuple[int, int]:
    """
    Suggest an approximate square image size for a target attention-memory budget.

    This inverts the same approximate quadratic attention formula used by
    estimate_attention_memory_usage() and should be treated as a sizing heuristic,
    not a guarantee of end-to-end inference memory fit.

    Args:
        target_memory_gb: Target attention-related memory in GB
        patch_size: Vision patch size in pixels
        extra_tokens: Non-image tokens added to the sequence
        bytes_per_element: Bytes per tensor element
        overhead_factor: Multiplier for estimated intermediate buffers

    Returns:
        Tuple of (width, height)
    """
    if target_memory_gb <= 0:
        raise ValueError("target_memory_gb must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if extra_tokens < 0:
        raise ValueError("extra_tokens cannot be negative")
    if bytes_per_element <= 0 or overhead_factor <= 0:
        raise ValueError("bytes_per_element and overhead_factor must be positive")

    target_bytes = target_memory_gb * (1024 ** 3)
    seq_length = int(np.sqrt(target_bytes / (bytes_per_element * overhead_factor)))
    num_patches = max(1, seq_length - extra_tokens)

    total_pixels = num_patches * (patch_size ** 2)
    optimal_size = int(np.sqrt(total_pixels))

    # Align to patch size boundary.
    optimal_size = max(patch_size, (optimal_size // patch_size) * patch_size)

    return (optimal_size, optimal_size)


RECOMMENDED_SIZES = {
    "low_memory": (512, 512),
    "standard": (768, 768),
    "high_memory": (1024, 1024),
    "very_high_memory": (1536, 1536),
}