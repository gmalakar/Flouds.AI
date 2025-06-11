# =============================================================================
# File: setup.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import importlib
import os
import subprocess

import nltk

from app.config.config_loader import ConfigLoader
from app.logger import get_logger

# Load settings using AppSettingsLoader
APP_SETTINGS = ConfigLoader.get_app_settings()
logger = get_logger("setup")

# Ensure APP_SETTINGS.app.working_dir is set to an absolute path
if not getattr(APP_SETTINGS.app, "working_dir", None) or not os.path.isabs(
    APP_SETTINGS.app.working_dir
):
    APP_SETTINGS.app.working_dir = os.getcwd()
logger.info(f"Appsettings->Working Directory: {APP_SETTINGS.app.working_dir}")

# Ensure ONNX root path is absolute and under the working directory if relative
if APP_SETTINGS.onnx.rootpath and not os.path.isabs(APP_SETTINGS.onnx.rootpath):
    # If not absolute, join with current working directory
    APP_SETTINGS.onnx.rootpath = os.path.join(os.getcwd(), APP_SETTINGS.onnx.rootpath)

logger.info(f"Appsettings-> ONNX root: {APP_SETTINGS.onnx.rootpath}")

logger.info(f"Environment: {os.getenv('FLOUDS_API_ENV', 'Production')}")

# NLTK setup
try:
    nltk.data.find("corpora/stopwords")
    logger.info("NLTK stopwords corpus already downloaded.")
except LookupError:
    nltk.download("stopwords")
    logger.info("NLTK stopwords corpus downloaded successfully.")
