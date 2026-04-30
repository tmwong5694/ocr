from pathlib import Path
import torch
from shared.src.shared_utils.timer import timer
from transformers import AutoProcessor, AutoModelForImageTextToText
from loguru import logger
from ml_pipeline.utils.config import load_config

# Load configuration from YAML
_CONFIG_PATH = Path(__file__).parent.parent.parent / ".config" / "detect_config.yaml"
_CONFIG = load_config(_CONFIG_PATH)
_GLM_CONFIG = _CONFIG.get("glm_ocr", {})

# Configuration values from detect_config.yaml
MODEL_PATH = _GLM_CONFIG.get("model_path", "zai-org/GLM-OCR")
DEFAULT_IMAGE_PATH = _GLM_CONFIG.get("default_image_path", "ml/data/prescription/Training/training_words/0.png")
MAX_TOKENS = _GLM_CONFIG.get("max_tokens", 8192)
DEFAULT_PROMPT = _GLM_CONFIG.get("prompt", "Text Recognition:")

# Auto-detect device if set to "auto"
DEVICE = _GLM_CONFIG.get("device", "auto")
if DEVICE == "auto":
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class GLMOCRModel:
    def __init__(self, model_path: str = MODEL_PATH, device: str = DEVICE):
        """
        Initialize the GLM-OCR model.

        Args:
            model_path: Path or HuggingFace model ID
            device: Device to load model on (cpu, cuda, mps)
        """
        self.model_path = model_path
        self.device = device
        self.processor: AutoProcessor | None = None
        self.model: AutoModelForImageTextToText | None = None

    @timer
    def load_model(self) -> None:
        """Load processor and model with timing."""
        logger.info(f"Loading GLM-OCR model from {self.model_path}")
        self._load_processor()
        self._load_model_weights()
        logger.info(f"Model loaded on device: {self.device}")

    @timer
    def _load_processor(self) -> None:
        """Load the AutoProcessor."""
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        logger.debug("AutoProcessor loaded")

    @timer
    def _load_model_weights(self) -> None:
        """Load model weights."""
        self.model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype="auto",
            device_map=self.device,
        )
        logger.debug("Model weights loaded")

    @timer
    def recognize_text(
            self,
            image_path: str,
            prompt: str = None
    ) -> str:
        """
        Recognize text from an image.

        Args:
            image_path: Path to the image file
            prompt: Prompt for the model (defaults to config value)

        Returns:
            Recognized text

        Raises:
            ValueError: If model is not loaded
            FileNotFoundError: If image file does not exist
        """
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

    @timer
    def _generate(self, inputs: dict) -> str:
        """Generate text from model."""
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS
        )
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return output_text


def main():
    """Main entry point for GLM-OCR inference."""
    logger.info("Starting GLM-OCR inference")

    # Initialize model
    glm_model = GLMOCRModel()
    glm_model.load_model()

    # Run inference
    try:
        result = glm_model.recognize_text(DEFAULT_IMAGE_PATH)
        logger.info(f"Recognition result: {result}")
        print(result)
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

    pass


if __name__ == "__main__":
    main()