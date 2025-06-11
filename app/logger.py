# =============================================================================
# File: logger.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(
    name: str = "flouds",
    log_file: str = "app.log",
    log_dir: str = "logs",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,  # 5 MB, 3 backups
) -> logging.Logger:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if log_file is None:
        log_file = f"{name}.log"
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    level = (
        logging.DEBUG if os.getenv("FLOUDS_DEBUG_MODE", "0") == "1" else logging.INFO
    )
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Rotating file handler (auto-purge after max_bytes)
    if not any(
        isinstance(h, RotatingFileHandler)
        and h.baseFilename == os.path.abspath(log_path)
        for h in logger.handlers
    ):
        fh = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
