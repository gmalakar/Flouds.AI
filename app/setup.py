# =============================================================================
# File: setup.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os

import nltk

from app.app_init import APP_SETTINGS
from app.config.config_loader import ConfigLoader
from app.logger import get_logger

# Load settings using AppSettingsLoader
logger = get_logger("setup")

# Ensure APP_SETTINGS.app.working_dir is set to an absolute path
if not getattr(APP_SETTINGS.app, "working_dir", None) or not os.path.isabs(
    APP_SETTINGS.app.working_dir
):
    APP_SETTINGS.app.working_dir = os.getcwd()
logger.info(f"Appsettings->Working Directory: {APP_SETTINGS.app.working_dir}")

# Handle ONNX root path
# First check if it was set from environment variable in config_loader.py
env_onnx_root = os.getenv("FLOUDS_ONNX_ROOT")
if env_onnx_root:
    # Environment variable takes precedence - no need to make it absolute
    # as Docker container paths might be different from host paths
    logger.info(f"Using ONNX root from environment: {APP_SETTINGS.onnx.model_path}")
else:
    # Only make paths absolute if they weren't set from environment
    if APP_SETTINGS.onnx.model_path and not os.path.isabs(APP_SETTINGS.onnx.model_path):
        # If not absolute and not from env var, join with current working directory
        APP_SETTINGS.onnx.model_path = os.path.join(
            os.getcwd(), APP_SETTINGS.onnx.model_path
        )
        logger.info(
            f"Converted relative ONNX path to absolute: {APP_SETTINGS.onnx.model_path}"
        )

logger.info(f"Appsettings-> ONNX root: {APP_SETTINGS.onnx.model_path}")
logger.info(f"Environment: {os.getenv('FLOUDS_API_ENV', 'Production')}")
