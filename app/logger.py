import logging
import os


def get_logger(
    name: str = "flouds", log_file: str = "app.log", log_dir: str = "logs"
) -> logging.Logger:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if log_file is None:
        log_file = f"{name}.log"
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler
    if not any(
        isinstance(h, logging.FileHandler)
        and h.baseFilename == os.path.abspath(log_path)
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
