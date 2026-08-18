import logging
from pathlib import Path

def setup_logger(log_file: Path):
    """
    Configure and return the project logger.
    """

    logger = logging.getLogger("POT_GPD")

    # Avoid adding duplicate handlers if run multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger