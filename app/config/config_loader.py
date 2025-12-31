# =============================================================================
# File: config_loader.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import json
import os
import sys
from typing import Dict, Optional

from app.config.appsettings import AppSettings
from app.config.onnx_config import OnnxConfig
from app.exceptions import (
    CacheInvalidationError,
    InvalidConfigError,
    MissingConfigError,
)
from app.logger import get_logger
from app.utils.log_sanitizer import sanitize_for_log

logger = get_logger("config_loader")


class ConfigLoader:
    __onnx_config_cache: Optional[Dict[str, OnnxConfig]] = None
    __config_file_mtime: Optional[float] = None
    __appsettings: Optional[AppSettings] = None

    @staticmethod
    def get_app_settings() -> AppSettings:
        """
        Loads AppSettings from appsettings.json and environment-specific override in the same folder.
        Performs a deep merge for nested config sections.
        """
        data = ConfigLoader._load_config_data("appsettings.json", True)
        ConfigLoader.__appsettings = AppSettings(**data)
        # Set production mode
        ConfigLoader.__appsettings.app.is_production = os.getenv(
            "FLOUDS_API_ENV", "Enterprise"
        ).lower() in ["production", "enterprise"]
        # set ONNX_ROOT
        # ONNX paths (allow None for development)
        ConfigLoader.__appsettings.onnx.onnx_path = os.getenv(
            "FLOUDS_ONNX_ROOT", ConfigLoader.__appsettings.onnx.onnx_path
        )
        ConfigLoader.__appsettings.onnx.config_file = os.getenv(
            "FLOUDS_ONNX_CONFIG_FILE", ConfigLoader.__appsettings.onnx.config_file
        )

        # Only require ONNX paths in production
        if ConfigLoader.__appsettings.app.is_production:
            if not ConfigLoader.__appsettings.onnx.onnx_path:
                logger.error(
                    "ONNX model path is required in production. Set FLOUDS_ONNX_ROOT environment variable."
                )
                sys.exit(1)
            if not ConfigLoader.__appsettings.onnx.config_file:
                logger.error(
                    "ONNX config file is required in production. Set FLOUDS_ONNX_CONFIG_FILE environment variable."
                )
                sys.exit(1)
        ConfigLoader.__appsettings.server.port = int(
            os.getenv("SERVER_PORT", ConfigLoader.__appsettings.server.port)
        )
        ConfigLoader.__appsettings.server.host = os.getenv(
            "SERVER_HOST", ConfigLoader.__appsettings.server.host
        )
        ConfigLoader.__appsettings.server.session_provider = os.getenv(
            "FLOUDS_MODEL_SESSION_PROVIDER",
            ConfigLoader.__appsettings.server.session_provider,
        )
        ConfigLoader.__appsettings.app.debug = (
            os.getenv(
                "APP_DEBUG_MODE", "1" if ConfigLoader.__appsettings.app.debug else "0"
            )
            == "1"
        )

        ConfigLoader.__appsettings.rate_limiting.enabled = (
            os.getenv(
                "FLOUDS_RATE_LIMIT_ENABLED",
                str(ConfigLoader.__appsettings.rate_limiting.enabled),
            ).lower()
            == "true"
        )
        ConfigLoader.__appsettings.rate_limiting.requests_per_minute = int(
            os.getenv(
                "FLOUDS_RATE_LIMIT_PER_MINUTE",
                ConfigLoader.__appsettings.rate_limiting.requests_per_minute,
            )
        )
        ConfigLoader.__appsettings.rate_limiting.requests_per_hour = int(
            os.getenv(
                "FLOUDS_RATE_LIMIT_PER_HOUR",
                ConfigLoader.__appsettings.rate_limiting.requests_per_hour,
            )
        )
        ConfigLoader.__appsettings.onnx.model_cache_size = int(
            os.getenv(
                "FLOUDS_MODEL_CACHE_SIZE",
                ConfigLoader.__appsettings.onnx.model_cache_size,
            )
        )
        ConfigLoader.__appsettings.app.max_request_size = int(
            os.getenv(
                "FLOUDS_MAX_REQUEST_SIZE",
                ConfigLoader.__appsettings.app.max_request_size,
            )
        )
        ConfigLoader.__appsettings.app.request_timeout = int(
            os.getenv(
                "FLOUDS_REQUEST_TIMEOUT", ConfigLoader.__appsettings.app.request_timeout
            )
        )

        # Parse CORS origins from environment
        cors_origins = os.getenv("FLOUDS_CORS_ORIGINS")
        if cors_origins:
            ConfigLoader.__appsettings.app.cors_origins = [
                origin.strip() for origin in cors_origins.split(",")
            ]

        # Security settings
        ConfigLoader.__appsettings.security.enabled = (
            os.getenv(
                "FLOUDS_SECURITY_ENABLED",
                str(ConfigLoader.__appsettings.security.enabled),
            ).lower()
            == "true"
        )

        # Monitoring settings
        ConfigLoader.__appsettings.monitoring.enable_metrics = (
            os.getenv(
                "FLOUDS_ENABLE_METRICS",
                str(ConfigLoader.__appsettings.monitoring.enable_metrics),
            ).lower()
            == "true"
        )
        ConfigLoader.__appsettings.monitoring.memory_threshold_mb = int(
            os.getenv(
                "FLOUDS_MEMORY_THRESHOLD_MB",
                ConfigLoader.__appsettings.monitoring.memory_threshold_mb,
            )
        )
        ConfigLoader.__appsettings.monitoring.cpu_threshold_percent = int(
            os.getenv(
                "FLOUDS_CPU_THRESHOLD_PERCENT",
                ConfigLoader.__appsettings.monitoring.cpu_threshold_percent,
            )
        )

        ConfigLoader.__appsettings.monitoring.cache_cleanup_max_age_seconds = int(
            os.getenv(
                "FLOUDS_CACHE_CLEANUP_MAX_AGE_SECONDS",
                ConfigLoader.__appsettings.monitoring.cache_cleanup_max_age_seconds,
            )
        )
        # Clients database path
        ConfigLoader.__appsettings.security.clients_db_path = os.getenv(
            "FLOUDS_CLIENTS_DB", ConfigLoader.__appsettings.security.clients_db_path
        )

        # Validate and create critical paths
        ConfigLoader._validate_paths()

        # logger.debug("Loaded AppSettings: %s", ConfigLoader.__appsettings)
        return ConfigLoader.__appsettings

    @staticmethod
    def _validate_paths():
        """Validate and create critical directories."""
        settings = ConfigLoader.__appsettings

        # Only validate ONNX paths in production
        if settings.app.is_production:
            # Validate ONNX root path
            if settings.onnx.onnx_path:
                if not os.path.exists(settings.onnx.onnx_path):
                    logger.error(
                        f"ONNX root path does not exist: {settings.onnx.onnx_path}"
                    )
                    sys.exit(1)
                if not os.path.isdir(settings.onnx.onnx_path):
                    logger.error(
                        f"ONNX root path is not a directory: {settings.onnx.onnx_path}"
                    )
                    sys.exit(1)
                logger.info(f"Validated ONNX root path: {settings.onnx.onnx_path}")

            # Validate ONNX config file
            if settings.onnx.config_file:
                if not os.path.exists(settings.onnx.config_file):
                    logger.error(
                        f"ONNX config file does not exist: {settings.onnx.config_file}"
                    )
                    sys.exit(1)
                if not os.path.isfile(settings.onnx.config_file):
                    logger.error(
                        f"ONNX config file is not a file: {settings.onnx.config_file}"
                    )
                    sys.exit(1)
                logger.info(f"Validated ONNX config file: {settings.onnx.config_file}")
        else:
            logger.info("Development mode: Skipping ONNX path validation")

        # Create and validate clients database directory
        if settings.security.clients_db_path:
            db_dir = os.path.dirname(os.path.abspath(settings.security.clients_db_path))
            if db_dir and not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                    logger.info(f"Created clients database directory: {db_dir}")
                except (OSError, PermissionError) as e:
                    logger.error(
                        "Failed to create clients database directory %s: %s",
                        sanitize_for_log(db_dir),
                        sanitize_for_log(str(e)),
                    )
                    sys.exit(1)
                except Exception as e:
                    logger.error(
                        "Unexpected error creating clients database directory %s: %s",
                        sanitize_for_log(db_dir),
                        sanitize_for_log(str(e)),
                    )
                    sys.exit(1)
            logger.info(
                f"Validated clients database path: {settings.security.clients_db_path}"
            )

        # Create log directory if specified
        log_path = os.getenv("FLOUDS_LOG_PATH")
        if log_path:
            if not os.path.exists(log_path):
                try:
                    os.makedirs(log_path, exist_ok=True)
                    logger.info(f"Created log directory: {log_path}")
                except (OSError, PermissionError) as e:
                    logger.error(
                        "Failed to create log directory %s: %s",
                        sanitize_for_log(log_path),
                        sanitize_for_log(str(e)),
                    )
                    sys.exit(1)
                except Exception as e:
                    logger.error(
                        "Unexpected error creating log directory %s: %s",
                        sanitize_for_log(log_path),
                        sanitize_for_log(str(e)),
                    )
                    sys.exit(1)
            elif not os.path.isdir(log_path):
                logger.error(f"Log path is not a directory: {log_path}")
                sys.exit(1)
            logger.info(f"Validated log directory: {log_path}")

    @staticmethod
    def get_onnx_config(key: str) -> OnnxConfig:
        """
        Loads OnnxConfig with caching and automatic cache invalidation.
        Cache is invalidated when config file is modified.
        """
        config_file_name = ConfigLoader.__appsettings.onnx.config_file

        # Check if cache needs refresh
        if ConfigLoader._should_refresh_cache(config_file_name):
            ConfigLoader._refresh_onnx_cache(config_file_name)

        if key not in ConfigLoader.__onnx_config_cache:
            raise MissingConfigError(
                f"Model config '{key}' not found in onnx_config.json"
            )
        return ConfigLoader.__onnx_config_cache[key]

    @staticmethod
    def _should_refresh_cache(config_file_name: str) -> bool:
        """Check if cache should be refreshed based on file modification time."""
        if ConfigLoader.__onnx_config_cache is None:
            return True

        try:
            current_mtime = os.path.getmtime(config_file_name)
            return ConfigLoader.__config_file_mtime != current_mtime
        except (OSError, FileNotFoundError):
            return True

    @staticmethod
    def _refresh_onnx_cache(config_file_name: str):
        """Refresh the ONNX configuration cache."""
        try:
            data = ConfigLoader._load_config_data(config_file_name)
            # Filter out documentation/metadata keys that start with underscore
            ConfigLoader.__onnx_config_cache = {
                k: OnnxConfig(**v) for k, v in data.items() if not k.startswith("_")
            }
            ConfigLoader.__config_file_mtime = os.path.getmtime(config_file_name)
            logger.debug(
                f"Refreshed ONNX config cache with {len(ConfigLoader.__onnx_config_cache)} models"
            )
        except (OSError, FileNotFoundError) as e:
            logger.error(f"ONNX config file not accessible: {e}")
            raise MissingConfigError(f"Cannot access ONNX config file: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid ONNX config format: {e}")
            raise InvalidConfigError(f"ONNX config file format error: {e}")
        except Exception as e:
            logger.error(f"Failed to refresh ONNX config cache: {e}")
            raise CacheInvalidationError(f"Cannot refresh config cache: {e}")

    @staticmethod
    def _load_config_data(config_file_name: str, check_env_file: bool = False) -> dict:
        """
        Loads a config file and merges with environment-specific override if present.
        Performs a deep merge for nested config sections.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, config_file_name)

        logger.debug(f"Loading config from {config_file_name}")

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Merge environment-specific config if requested and it exists (deep merge)
        if check_env_file:
            env = os.getenv("FLOUDS_API_ENV", "Production")
            name, ext = os.path.splitext(config_file_name)
            env_file = f"{name}.{env.lower()}{ext}"
            env_path = os.path.join(base_dir, env_file)
            logger.debug(f"Loading config from {env_file}")
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    env_data = json.load(f)
                deep_update(data, env_data)
            except (OSError, FileNotFoundError):
                logger.warning(
                    f"Environment-specific config file not found: {env_file}. Using base config."
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(
                    "Invalid environment config format in %s: %s",
                    sanitize_for_log(env_file),
                    sanitize_for_log(str(e)),
                )
                raise InvalidConfigError(f"Environment config format error: {e}")
            except Exception as e:
                logger.error(
                    "Unexpected error loading environment config %s: %s",
                    sanitize_for_log(env_file),
                    sanitize_for_log(str(e)),
                )
                raise InvalidConfigError(f"Cannot load environment config: {e}")

        return data

    @staticmethod
    def clear_cache():
        """Clear all configuration caches."""
        ConfigLoader.__onnx_config_cache = None
        ConfigLoader.__config_file_mtime = None
        logger.info("Configuration cache cleared")

    @staticmethod
    def get_cache_stats() -> Dict[str, any]:
        """Get cache statistics for monitoring."""
        return {
            "onnx_configs_cached": (
                len(ConfigLoader.__onnx_config_cache)
                if ConfigLoader.__onnx_config_cache
                else 0
            ),
            "cache_file_mtime": ConfigLoader.__config_file_mtime,
            "cache_loaded": ConfigLoader.__onnx_config_cache is not None,
        }


# Example usage:
# settings = ConfigLoader.get_app_settings()
# onnx_cfg = ConfigLoader.get_onnx_config()
