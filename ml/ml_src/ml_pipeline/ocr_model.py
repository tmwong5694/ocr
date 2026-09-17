from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal

import pymupdf
import torch
from PIL import Image
from loguru import logger
from transformers import AutoProcessor, AutoModelForImageTextToText

from ml_pipeline.utils.config import load_config
from ml_pipeline.utils.image_preprocessing import estimate_attention_memory_usage


# Load configuration from YAML
_CONFIG_PATH = Path(__file__).parent.parent.parent / ".config" / "detect_config.yaml"
_CONFIG = load_config(_CONFIG_PATH)
_OCR_CONFIG = _CONFIG.get("ocr_model", {})

# Configuration values from detect_config.yaml
MODEL_NAME = _OCR_CONFIG.get("model_name", "zai-org/GLM-OCR")
DEFAULT_IMAGE_PATH = _OCR_CONFIG.get("default_image_path", "ml/data/prescription/Training/training_words/0.png")
MAX_TOKENS = _OCR_CONFIG.get("max_tokens", 8192)


def _load_default_prompt():
    """Load the OCR prompt from a dedicated text file."""
    prompt_path = Path(__file__).parent.parent.parent / ".config" / "ocr_prompt.txt"
    try:
        with open(prompt_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found at {prompt_path}, using fallback")
        return "Recognize text in the image."


DEFAULT_PROMPT = _load_default_prompt()

# Auto-detect device if set to "auto"
DEVICE = _OCR_CONFIG.get("device", "auto")
if DEVICE == "auto":
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ImageInput = str | Path | bytes | BinaryIO | Image.Image


class OCRModel:
    """
    Flexible vision-language model for OCR and image-to-text tasks.

    Supports any HuggingFace model that uses AutoProcessor and AutoModelForImageTextToText.
    Examples: GLM-OCR, Qwen-VL, LLaVA, etc.

    Easy model switching via:
    1. Config file: update 'model_name' in detect_config.yaml
    2. Constructor: pass different model_name during instantiation
    3. Method parameter: specify model_name when calling recognize_text()
    4. switch_model() method: dynamically switch between models
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Literal["cuda", "mps", "cpu"] = DEVICE,
        model_kwargs: dict = None,
        enable_preprocessing: bool = True,
        max_image_width: int = None,
        max_image_height: int = None,
    ):
        """
        Initialize the vision-language OCR model.

        Args:
            model_name: HuggingFace model ID or path (e.g., "zai-org/GLM-OCR", "Qwen/Qwen-VL")
            device: Device to load model on (cpu, cuda, mps, auto)
            model_kwargs: Additional kwargs for model.from_pretrained()
            enable_preprocessing: Whether to automatically resize large images (default: True)
            max_image_width: Maximum image width in pixels (default: from config or 768)
            max_image_height: Maximum image height in pixels (default: from config or 768)
        """
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.processor: AutoProcessor | None = None
        self.model: AutoModelForImageTextToText | None = None

        self.enable_preprocessing = enable_preprocessing
        preprocess_config = _OCR_CONFIG.get("preprocess", {})
        self.max_image_width = max_image_width or preprocess_config.get("max_width", 768)
        self.max_image_height = max_image_height or preprocess_config.get("max_height", 768)
        self.quality = preprocess_config.get("quality", 85)

        logger.debug(f"Initialized OCR model for: {self.model_name}")
        logger.debug(
            f"Image preprocessing: {self.enable_preprocessing} "
            f"(max size: {self.max_image_width}x{self.max_image_height})"
        )

    def load_model(self, model_name: str = None) -> None:
        """Load processor and model explicitly."""
        if model_name:
            self.model_name = model_name

        logger.info(f"Loading model: {self.model_name}")

        logger.debug("Loading processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            logger.debug("Processor loaded")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise

        logger.debug("Loading model weights...")
        try:
            dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
            self.model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                dtype=dtype,
                device_map=self.device,
                **self.model_kwargs,
            )
            logger.debug("Model loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        logger.info(f"Model loaded on device: {self.device}")

    def image_to_text(
        self,
        image: ImageInput,
        prompt: str = "",
        model_name: str = None,
    ) -> str:
        """
        Recognize text from an image using the current or specified model.

        Args:
            image: Path, bytes, file-like object, or PIL image
            prompt: Prompt for the model (defaults to config value)
            model_name: Switch to a different model for this inference

        Returns:
            Recognized text
        """
        if model_name and model_name != self.model_name:
            logger.info(f"Switching model from {self.model_name} to {model_name}")
            self.load_model(model_name)

        if self.model is None or self.processor is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before recognize_text(). "
                "Example: ocr_model.load_model()"
            )

        if prompt == "":
            prompt = DEFAULT_PROMPT

        image_obj = self._load_image(image)

        if self.enable_preprocessing:
            image_obj = self._preprocess_image(image_obj)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_obj},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._prepare_inputs(messages)
        return self._generate(inputs)

    def check_contains_text(self, pdf_path: str):
        has_any_text = False
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                if page.get_text().strip():
                    has_any_text = True
                    break

        return has_any_text

    def extract_image(self, pdf_path: str):
        extracted = []
        with pymupdf.open(pdf_path) as doc:
            for page_idx, _page in enumerate(doc):
                for img_index, img in enumerate(doc.get_page_images(page_idx)):
                    xref = img[0]
                    image_data = doc.extract_image(xref)
                    ext = image_data["ext"]
                    out_path = Path(f"page{page_idx}_img{img_index}.{ext}")

                    with open(out_path, "wb") as writer:
                        writer.write(image_data["image"])

                    extracted.append(str(out_path))

        return extracted

    def _load_image(self, image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.copy().convert("RGB")

        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                return img.convert("RGB")

        if isinstance(image, bytes):
            with Image.open(BytesIO(image)) as img:
                return img.convert("RGB")

        if hasattr(image, "seek"):
            try:
                image.seek(0)
            except Exception:
                pass
            with Image.open(image) as img:
                return img.convert("RGB")

        raise TypeError(f"Unsupported image input type: {type(image)!r}")

    def _prepare_inputs(self, messages: list) -> dict:
        """Prepare model inputs from messages."""
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            inputs.pop("token_type_ids", None)
            logger.debug("Inputs prepared")
            return inputs
        except Exception as e:
            logger.error(f"Failed to prepare inputs: {e}")
            raise

    def _generate(self, inputs: dict, max_tokens: int = None) -> str:
        """
        Generate text from model.

        Args:
            inputs: Prepared model inputs
            max_tokens: Override default max_tokens
        """
        try:
            tokens = max_tokens or MAX_TOKENS
            generated_ids = self.model.generate(**inputs, max_new_tokens=tokens)
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return output_text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to reduce memory usage.

        Resizes large images while maintaining aspect ratio to prevent
        the attention mechanism from allocating excessive memory.
        """
        try:
            working = image.copy()
            if working.mode != "RGB":
                working = working.convert("RGB")

            width, height = working.size
            memory_gb = estimate_attention_memory_usage(width, height)

            logger.info(f"Image dimensions: {width}x{height}")
            logger.info(f"Estimated attention memory: {memory_gb:.2f} GB")

            if width > self.max_image_width or height > self.max_image_height:
                logger.warning(
                    f"Image size ({width}x{height}) exceeds max ({self.max_image_width}x{self.max_image_height}). "
                    f"Resizing to reduce memory usage..."
                )

                working.thumbnail(
                    (self.max_image_width, self.max_image_height),
                    Image.Resampling.LANCZOS,
                )

                new_width, new_height = working.size
                new_memory_gb = estimate_attention_memory_usage(new_width, new_height)
                logger.info(f"After preprocessing: {new_width}x{new_height} (~{new_memory_gb:.2f} GB)")

            else:
                logger.debug(f"Image size is acceptable ({width}x{height})")

            return working

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            logger.warning("Attempting to use original image anyway...")
            return image