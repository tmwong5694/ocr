import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def set_logger(name: str, level: str, logger_path: str | Path, use_tqdm_handler: bool = False) -> logging.Logger:

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARN,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    logger.setLevel(logging_levels[level])

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)-8s - %(filename)s - %(funcName)s():%(lineno)d - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if use_tqdm_handler:
        console_handler = TqdmLoggingHandler()
    else:
        console_handler = logging.StreamHandler()

    console_handler.setLevel(logging_levels[level])
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger_path = Path(logger_path)
    if not logger_path.parent.exists():
        logger_path.parent.mkdir(exist_ok=True, parents=True)
    file_handler = TimedRotatingFileHandler(logger_path, encoding="utf-8", when="midnight", interval=1)
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)    

    return logger

if __name__ == "__main__":

    logger = set_logger(__name__, "info", logger_path=Path("logs") / "testing.log")
    logger.info("Set up success!")
    pass