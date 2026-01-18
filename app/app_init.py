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

# Development mode handling
api_env = os.environ.get("FLOUDS_API_ENV", "Enterprise").lower()
is_production = api_env in ["production", "enterprise"]

if not is_production:
    logger.info("Development mode: ONNX paths optional")
else:
    logger.info("Production mode: ONNX paths required")

logger.info(f"FLOUDS_API_ENV: {os.environ.get('FLOUDS_API_ENV', 'Enterprise')}")
logger.info(f"FLOUDS_ONNX_ROOT: {os.environ.get('FLOUDS_ONNX_ROOT', 'Not set')}")
logger.info(f"FLOUDS_ONNX_CONFIG_FILE: {os.environ.get('FLOUDS_ONNX_CONFIG_FILE', 'Not set')}")
logger.info(f"FLOUDS_CLIENTS_DB: {os.environ.get('FLOUDS_CLIENTS_DB', 'clients.db')}")
logger.info(f"FLOUDS_LOG_PATH: {os.environ.get('FLOUDS_LOG_PATH', 'Not set')}")

# Load application settings with path validation
APP_SETTINGS = ConfigLoader.get_app_settings()

# Log final configuration
logger.info(f"Production mode: {APP_SETTINGS.app.is_production}")
logger.info(f"Security enabled: {APP_SETTINGS.security.enabled}")
logger.info(f"Clients DB path: {APP_SETTINGS.security.clients_db_path}")
if APP_SETTINGS.onnx.onnx_path:
    logger.info(f"ONNX root validated: {APP_SETTINGS.onnx.onnx_path}")
if APP_SETTINGS.onnx.config_file:
    logger.info(f"ONNX config validated: {APP_SETTINGS.onnx.config_file}")
# NLTK setup
try:
    nltk.data.find("tokenizers/punkt_tab")
    logger.info("NLTK punkt_tab tokenizer already downloaded.")
except (LookupError, OSError):
    try:
        nltk.data.find("tokenizers/punkt")
        logger.info("NLTK punkt tokenizer already downloaded.")
    except (LookupError, OSError):
        try:
            nltk.download("punkt_tab")
            logger.info("NLTK punkt_tab tokenizer downloaded successfully.")
        except Exception:
            nltk.download("punkt")
            logger.info("NLTK punkt tokenizer downloaded successfully.")

logger.info("Application initialization completed successfully")
