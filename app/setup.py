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

logger.info(f"Environment: {os.getenv('FLOUDS_API_ENV', 'Production')}")


def ensure_server_installed_and_imported(server_name):
    """Ensure the required ASGI server is installed and import it."""
    try:
        return importlib.import_module(server_name)
    except ImportError:
        logger.warning(f"{server_name} not found. Installing...")
        try:
            subprocess.run(["pip", "install", server_name], check=True)
            logger.info(f"{server_name} installed successfully.")
            return importlib.import_module(server_name)
        except Exception as e:
            logger.error(f"Failed to install or import {server_name}: {e}")
            return None


SERVER_MODULE = ensure_server_installed_and_imported(APP_SETTINGS.server.type.lower())

# NLTK setup
try:
    nltk.data.find("corpora/stopwords")
    logger.info("NLTK stopwords corpus already downloaded.")
except LookupError:
    nltk.download("stopwords")
    logger.info("NLTK stopwords corpus downloaded successfully.")
