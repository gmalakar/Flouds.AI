# =============================================================================
# File: app_init.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
import os

import nltk

from app.config.config_loader import ConfigLoader
from app.logger import get_logger

logger = get_logger("app_init")

# Set environment variables to None if not production
if os.environ.get("FLOUDS_API_ENV", "Production").lower() != "production":
    os.environ.pop("FLOUDS_ONNX_ROOT", None)
    os.environ.pop("FLOUDS_ONNX_CONFIG_FILE", None)
    os.environ.pop("FLOUDS_LOG_PATH", None)
    logger.info("Development mode: ONNX and log paths set to None")

logger.info(f"FLOUDS_API_ENV: {os.environ.get('FLOUDS_API_ENV')}")
logger.info(f"FLOUDS_ONNX_ROOT: {os.environ.get('FLOUDS_ONNX_ROOT')}")
logger.info(f"FLOUDS_ONNX_CONFIG_FILE: {os.environ.get('FLOUDS_ONNX_CONFIG_FILE')}")

APP_SETTINGS = ConfigLoader.get_app_settings()

# NLTK setup
try:
    nltk.data.find("corpora/stopwords")
    logger.info("NLTK stopwords corpus already downloaded.")
except LookupError:
    nltk.download("stopwords")
    logger.info("NLTK stopwords corpus downloaded successfully.")
