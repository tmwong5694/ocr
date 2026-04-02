import logging
from pathlib import Path

def set_logger(name: str, level: str, logger_path: str | Path) -> logging.Logger:

    logger = logging.getLogger(name)

    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARN,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    logger.setLevel(logging_levels[level])

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    logger_path = Path(logger_path)
    if not logger_path.parent.exists():
        logger_path.parent.mkdir(exist_ok=True, parents=True)
    file_handler = logging.FileHandler(logger_path, encoding="utf-8")

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    console_handler.setFormatter(formatter)

    return logger

if __name__ == "__main__":

    logg = set_logger(__name__, "info", logger_path=Path("logs") / "testing.log")
    logg.debug("Set up success!")
    pass