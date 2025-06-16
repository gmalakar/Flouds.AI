# =============================================================================
# File: base_nlp_service.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
import nltk

from app.config.config_loader import ConfigLoader
from app.logger import get_logger

logger = get_logger("app_init")

APP_SETTINGS = ConfigLoader.get_app_settings()

# NLTK setup
try:
    nltk.data.find("corpora/stopwords")
    logger.info("NLTK stopwords corpus already downloaded.")
except LookupError:
    nltk.download("stopwords")
    logger.info("NLTK stopwords corpus downloaded successfully.")
