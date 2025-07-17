# =============================================================================
# File: logger.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str = "flouds") -> logging.Logger:
    is_production = os.getenv("FLOUDS_API_ENV", "Production").lower() == "production"

    if is_production:
        log_dir = os.getenv("FLOUDS_LOG_PATH", "/flouds-ai/logs")
    else:
        # Get parent directory of app folder and create logs folder there
        current_dir = os.path.dirname(os.path.abspath(__file__))  # app folder
        parent_dir = os.path.dirname(current_dir)  # parent of app
        log_dir = os.path.join(parent_dir, "logs")

    # Add date to log file name
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = f"flouds-ai-{date_str}.log"
    max_bytes = 10485760
    backup_count = 5
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
        try:
            fh = RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except PermissionError:
            print(f"Warning: Permission denied for log file: {log_path}")
            logger.warning(f"Cannot write to log file: {log_path} - permission denied")

    return logger
