from pathlib import Path
import fitz
import torch
from shared_utils.timer import timer
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import Literal
from loguru import logger
from ml_pipeline.utils.config import load_config
from ml_pipeline.utils.image_preprocessing import (
    resize_image_for_ocr,
    estimate_attention_memory_usage,
    get_image_dimensions,
)

# Load configuration from YAML
_CONFIG_PATH = Path(__file__).parent.parent.parent / ".config" / "detect_config.yaml"
_CONFIG = load_config(_CONFIG_PATH)
_OCR_CONFIG = _CONFIG.get("ocr_model", {})

# Configuration values from detect_config.yaml
MODEL_NAME = _OCR_CONFIG.get("model_name", "zai-org/GLM-OCR")
DEFAULT_IMAGE_PATH = _OCR_CONFIG.get("default_image_path", "ml/data/prescription/Training/training_words/0.png")
MAX_TOKENS = _OCR_CONFIG.get("max_tokens", 8192)
DEFAULT_PROMPT = _OCR_CONFIG.get("prompt", "Text Recognition:")

# Auto-detect device if set to "auto"
DEVICE = _OCR_CONFIG.get("device", "auto")
if DEVICE == "auto":
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


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

        # Image preprocessing configuration
        self.enable_preprocessing = enable_preprocessing
        preprocess_config = _OCR_CONFIG.get("preprocess", {})
        self.max_image_width = max_image_width or preprocess_config.get("max_width", 768)
        self.max_image_height = max_image_height or preprocess_config.get("max_height", 768)
        self.quality = preprocess_config.get("quality", 85)
        
        logger.debug(f"Initialized OCR model for: {self.model_name}")
        logger.debug(f"Image preprocessing: {self.enable_preprocessing} "
                     f"(max size: {self.max_image_width}x{self.max_image_height})")

    @timer
    def load_model(self, model_name: str = None) -> None:
        """Load processor and model explicitly."""
        if model_name:
            self.model_name = model_name

        logger.info(f"Loading model: {self.model_name}")

        # Load processor
        logger.debug("Loading processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            logger.debug(f"Processor loaded")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise

        # Load model
        logger.debug("Loading model weights...")
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                dtype=torch.float16,
                device_map=self.device,
                **self.model_kwargs
            )
            logger.debug(f"Model loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        logger.info(f"Model loaded on device: {self.device}")


    @timer
    def image_to_text(
            self,
            image_path: str,
            prompt: str = None,
            model_name: str = None
    ) -> str:
        """
        Recognize text from an image using the current or specified model.

        Args:
            image_path: Path to the image file
            prompt: Prompt for the model (defaults to config value)
            model_name: Switch to a different model for this inference

        Returns:
            Recognized text

        Raises:
            ValueError: If model is not loaded
            FileNotFoundError: If image file does not exist
        """
        # Switch model if requested
        if model_name and model_name != self.model_name:
            logger.info(f"Switching model from {self.model_name} to {model_name}")
            self.load_model(model_name)

        if self.model is None or self.processor is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before recognize_text(). "
                "Example: ocr_model.load_model()"
            )

        if prompt is None:
            prompt = DEFAULT_PROMPT

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Preprocess image if enabled
        if self.enable_preprocessing:
            image_path = self._preprocess_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        inputs = self._prepare_inputs(messages)
        output_text = self._generate(inputs)

        return output_text

    def check_contains_text(self, pdf_path: str):

        has_any_text = False
        with fitz.open(pdf_path) as doc:
            for page in doc:
                if page.get_text().strip():
                    has_any_text = True
                    break

        return has_any_text


    def pdf_to_text(self, pdf_path: str):

        doc = fitz.open(pdf_path)
        for page_idx, page in enumerate(doc):
            for img_index, img in enumerate(doc.get_page_images(page_idx)):
                xref = img[0]
                image_data = doc.extract_image(xref)

                with open(f"page{page_idx}_img{img_index}.{image_data["ext"]}", "wb") as writer:
                    writer.write(image_data["image"])

        return

    @timer
    def _prepare_inputs(self, messages: list) -> dict:
        """Prepare model inputs from messages."""
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            inputs.pop("token_type_ids", None)
            logger.debug("Inputs prepared")
            return inputs
        except Exception as e:
            logger.error(f"Failed to prepare inputs: {e}")
            raise

    @timer
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
                skip_special_tokens=True
            )
            return output_text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    @timer
    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image to reduce memory usage.

        Resizes large images while maintaining aspect ratio to prevent
        the attention mechanism from allocating excessive memory.

        Args:
            image_path: Path to the image file
            
        Returns:
            Path to the preprocessed image (may be same as input if no resize needed)
        """
        try:
            width, height = get_image_dimensions(image_path)
            memory_gb = estimate_attention_memory_usage(width, height)

            logger.info(f"Image dimensions: {width}x{height}")
            logger.info(f"Estimated attention memory: {memory_gb:.2f} GB")

            # Check if resizing is needed
            if width > self.max_image_width or height > self.max_image_height:
                logger.warning(
                    f"Image size ({width}x{height}) exceeds max ({self.max_image_width}x{self.max_image_height}). "
                    f"Resizing to reduce memory usage..."
                )
                
                preprocessed_path = resize_image_for_ocr(
                    image_path,
                    max_width=self.max_image_width,
                    max_height=self.max_image_height,
                    quality=self.quality,
                )
                
                # Check memory of resized image
                new_width, new_height = get_image_dimensions(preprocessed_path)
                new_memory_gb = estimate_attention_memory_usage(new_width, new_height)
                logger.info(f"After preprocessing: {new_width}x{new_height} (~{new_memory_gb:.2f} GB)")

                return preprocessed_path
            else:
                logger.debug(f"Image size is acceptable ({width}x{height})")
                return image_path

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            logger.warning("Attempting to use original image anyway...")
            return image_path
