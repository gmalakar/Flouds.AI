# =============================================================================
# File: app_init.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from app.config.config_loader import ConfigLoader

APP_SETTINGS = ConfigLoader.get_app_settings()
