import sys
from loguru import logger
from pathlib import Path
from tqdm import tqdm


def set_loguru(level: str, logger_path: str | Path) -> None:
    """Configures Loguru to work with tqdm and midnight file rotation."""
    logger_path = Path(logger_path)
    logger_path.parent.mkdir(exist_ok=True, parents=True)

    # Remove default terminal output
    logger.remove()

    logger.add(
        lambda msg: tqdm.write(msg, end=""), 
        level=level.upper(), 
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:{line} - <level>{message}</level>"
    )

    logger.add(
        str(logger_path),
        level=level.upper(),
        rotation="00:00",      # Rotates at midnight automatically
        retention="30 days",   # Automatically cleans up old logs
        format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level: <8} - {file} - {function}():{line} - {message}"
    )