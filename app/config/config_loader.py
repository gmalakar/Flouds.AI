# =============================================================================
# File: config_loader.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import json
import os
import sys

from app.config.appsettings import AppSettings
from app.config.onnx_config import OnnxConfig
from app.logger import get_logger

logger = get_logger("config_loader")


class ConfigLoader:
    __onnx_config_cache = None
    __appsettings = None

    @staticmethod
    def get_app_settings() -> AppSettings:
        """
        Loads AppSettings from appsettings.json and environment-specific override in the same folder.
        Performs a deep merge for nested config sections.
        """
        data = ConfigLoader._load_config_data("appsettings.json", True)
        ConfigLoader.__appsettings = AppSettings(**data)
        # set isproduction
        ConfigLoader.__appsettings.app.is_production = (
            os.getenv("FLOUDS_API_ENV", "Production").lower() == "production"
        )
        # set ONNX_ROOT
        ConfigLoader.__appsettings.onnx.onnx_path = os.getenv(
            "FLOUDS_ONNX_ROOT", ConfigLoader.__appsettings.onnx.onnx_path
        )
        if not ConfigLoader.__appsettings.onnx.onnx_path:
            logger.error(
                f"ONNX model path is not set. Please set it in appsettings.json or via the FLOUDS_ONNX_ROOT environment variable and restart application."
            )
            sys.exit(1)
        ConfigLoader.__appsettings.onnx.config_file = os.getenv(
            "FLOUDS_ONNX_CONFIG_FILE", ConfigLoader.__appsettings.onnx.config_file
        )
        if not ConfigLoader.__appsettings.onnx.config_file:
            logger.error(
                "ONNX config file is not set. Please set it in appsettings.json or via the FLOUDS_ONNX_CONFIG_FILE environment variable and restart application."
            )
            sys.exit(1)
        ConfigLoader.__appsettings.server.port = int(
            os.getenv("FLOUDS_PORT", ConfigLoader.__appsettings.server.port)
        )
        ConfigLoader.__appsettings.server.host = os.getenv(
            "FLOUDS_HOST", ConfigLoader.__appsettings.server.host
        )
        ConfigLoader.__appsettings.server.type = os.getenv(
            "FLOUDS_SERVER_TYPE", ConfigLoader.__appsettings.server.type
        )
        ConfigLoader.__appsettings.server.session_provider = os.getenv(
            "FLOUDS_MODEL_SESSION_PROVIDER",
            ConfigLoader.__appsettings.server.session_provider,
        )
        ConfigLoader.__appsettings.app.debug = (
            os.getenv("FLOUDS_DEBUG_MODE", "0") == "1"
        )
        # logger.debug("Loaded AppSettings: %s", ConfigLoader.__appsettings)
        return ConfigLoader.__appsettings

    @staticmethod
    def get_onnx_config(key: str) -> OnnxConfig:
        """
        Loads OnnxConfig from onnx_config.json and environment-specific override in the same folder.
        Performs a deep merge for nested config sections.
        Only loads from file if config is not in cache.
        Returns the OnnxConfig for the specified key/model.
        Raises KeyError if the key is not found.
        """
        if ConfigLoader.__onnx_config_cache is None:
            config_file_name = ConfigLoader.__appsettings.onnx.config_file
            data = ConfigLoader._load_config_data(config_file_name)
            ConfigLoader.__onnx_config_cache = {
                k: OnnxConfig(**v) for k, v in data.items()
            }

        if key not in ConfigLoader.__onnx_config_cache:
            raise KeyError(f"Model config '{key}' not found in onnx_config.json")
        return ConfigLoader.__onnx_config_cache[key]

    @staticmethod
    def _load_config_data(config_file_name: str, check_env_file: bool = False) -> dict:
        """
        Loads a config file and merges with environment-specific override if present.
        Performs a deep merge for nested config sections.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(base_dir, config_file_name)

        logger.debug(f"Loading config from {base_path}")

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Config file not found: {base_path}")

        with open(base_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Merge environment-specific config if requested and it exists (deep merge)
        if check_env_file:
            env = os.getenv("FLOUDS_API_ENV", "Production")
            name, ext = os.path.splitext(config_file_name)
            env_path = os.path.join(base_dir, f"{name}.{env.lower()}{ext}")
            logger.debug(f"Loading config from {env_path}")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    env_data = json.load(f)
                deep_update(data, env_data)
            else:
                logger.warning(
                    f"Environment-specific config file not found: {env_path}. Using base config."
                )

        return data


# Example usage:
# settings = ConfigLoader.get_app_settings()
# onnx_cfg = ConfigLoader.get_onnx_config()
