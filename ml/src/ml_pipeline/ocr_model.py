from pathlib import Path
import torch
from shared.src.shared_utils.timer import timer
from transformers import AutoProcessor, AutoModelForImageTextToText
from loguru import logger
from ml_pipeline.utils.config import load_config

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
        device: str = DEVICE,
        model_kwargs: dict = None
    ):
        """
        Initialize the vision-language OCR model.

        Args:
            model_name: HuggingFace model ID or path (e.g., "zai-org/GLM-OCR", "Qwen/Qwen-VL")
            device: Device to load model on (cpu, cuda, mps, auto)
            model_kwargs: Additional kwargs for model.from_pretrained()
        """
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.processor: AutoProcessor | None = None
        self.model: AutoModelForImageTextToText | None = None
        
        logger.debug(f"Initialized OCR model for: {self.model_name}")

    @timer
    def load_model(self, model_name: str = None) -> None:
        """
        Load processor and model with timing.
        
        Args:
            model_name: Override the model name (useful for testing different models)
        """
        if model_name:
            self.model_name = model_name
            
        logger.info(f"Loading model: {self.model_name}")
        self._load_processor()
        self._load_model_weights()
        logger.info(f"Model '{self.model_name}' loaded on device: {self.device}")

    @timer
    def _load_processor(self) -> None:
        """Load the AutoProcessor."""
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            logger.debug(f"Processor loaded for {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load processor for {self.model_name}: {e}")
            raise

    @timer
    def _load_model_weights(self) -> None:
        """Load model weights."""
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype="auto",
                device_map=self.device,
                **self.model_kwargs  # Allow custom kwargs per model
            )
            logger.debug(f"Model weights loaded for {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    @timer
    def recognize_text(
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
        
        if prompt is None:
            prompt = DEFAULT_PROMPT
            
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

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
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=tokens
            )
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return output_text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def switch_model(self, model_name: str) -> None:
        """Convenience method to switch to a different model."""
        self.load_model(model_name)
