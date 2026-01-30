# =============================================================================
# File: config_loader.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import json
import os
import sys
from typing import Any, Dict, Optional, cast

from app.config.appsettings import AppSettings
from app.config.onnx_config import OnnxConfig
from app.exceptions import CacheInvalidationError, InvalidConfigError, MissingConfigError
from app.logger import get_logger
from app.utils.log_sanitizer import sanitize_for_log

# `validate_safe_path` intentionally not used here; keep import for future path checks
# (If you prefer, remove the import to silence linter warnings.)
# from app.utils.path_validator import validate_safe_path

logger = get_logger("config_loader")


class ConfigLoader:
    __onnx_config_cache: Optional[Dict[str, OnnxConfig]] = None
    __config_file_mtime: Optional[float] = None
    __appsettings: Optional[AppSettings] = None

    @staticmethod
    def _getenv_first(*names: str) -> Optional[str]:
        """Return first non-empty environment variable value for given names."""
        for n in names:
            v = os.getenv(n)
            if v is not None:
                return v
        return None

    @staticmethod
    def _parse_bool(val: Optional[str]) -> Optional[bool]:
        if val is None:
            return None
        return str(val).lower() in ("1", "true", "yes")

    @staticmethod
    def _parse_int(val: Optional[str]) -> Optional[int]:
        if val is None:
            return None
        try:
            return int(val)
        except ValueError:
            return None

    @staticmethod
    def get_app_settings() -> AppSettings:
        """Load AppSettings from `appsettings.json`, apply environment overrides,
        and return a populated AppSettings instance.

        This follows the canonical pattern used in FloudsVector: it validates
        config file paths, supports an environment-specific override file, and
        applies a consistent set of environment variable overrides.
        """

        # Load base config with optional env-specific override (deep merge)
        data: Dict[str, Any] = ConfigLoader._load_config_data("appsettings.json", True)
        ConfigLoader.__appsettings = AppSettings(**data)

        # Ensure CSP values are taken from the JSON if present. We keep the
        # fields default as None in code and populate them explicitly so that the
        # runtime values come from `appsettings.json` rather than hard-coded
        # defaults in the model.
        try:
            sec_section = data.get("security") if isinstance(data, dict) else None
            if sec_section and isinstance(sec_section, dict):
                for k in ("csp_script_src", "csp_style_src", "csp_img_src", "csp_connect_src"):
                    if k in sec_section:
                        try:
                            setattr(
                                ConfigLoader.__appsettings.security,
                                k,
                                sec_section.get(k),
                            )
                        except Exception:
                            logger.debug(f"Failed to apply security.{k} from appsettings.json")
        except Exception:
            # Fail-safe: don't let CSP mapping break startup; log and continue
            logger.debug("No CSP values applied from appsettings.json")

        # Environment variable overrides for CSP arrays. Support either a
        # JSON array value or a comma-separated string. Recognize both a
        # single-underscore and double-underscore env var naming style.
        try:
            env_map = {
                "csp_script_src": (
                    "FLOUDS_SECURITY_CSP_SCRIPT_SRC",
                    "FLOUDS_SECURITY__CSP_SCRIPT_SRC",
                ),
                "csp_style_src": (
                    "FLOUDS_SECURITY_CSP_STYLE_SRC",
                    "FLOUDS_SECURITY__CSP_STYLE_SRC",
                ),
                "csp_img_src": ("FLOUDS_SECURITY_CSP_IMG_SRC", "FLOUDS_SECURITY__CSP_IMG_SRC"),
                "csp_connect_src": (
                    "FLOUDS_SECURITY_CSP_CONNECT_SRC",
                    "FLOUDS_SECURITY__CSP_CONNECT_SRC",
                ),
                "csp_font_src": (
                    "FLOUDS_SECURITY_CSP_FONT_SRC",
                    "FLOUDS_SECURITY__CSP_FONT_SRC",
                ),
                "csp_worker_src": (
                    "FLOUDS_SECURITY_CSP_WORKER_SRC",
                    "FLOUDS_SECURITY__CSP_WORKER_SRC",
                ),
            }
            for field, names in env_map.items():
                env_val = ConfigLoader._getenv_first(*names)
                if env_val is None:
                    continue
                parsed: Optional[list] = None
                # Try JSON first
                try:
                    maybe = json.loads(env_val)
                    if isinstance(maybe, list):
                        # Normalize items (preserve any intentional quoting such as 'self')
                        parsed = [str(x).strip() for x in maybe if str(x).strip()]
                except Exception:
                    # Fallback: comma-separated list. Accept values like
                    # ["'self'","https://..."] or 'self,https://...'
                    raw = env_val.strip()
                    if raw.startswith("[") and raw.endswith("]"):
                        raw = raw[1:-1]
                    parts = []
                    for p in raw.split(","):
                        s = p.strip()
                        if not s:
                            continue
                        # Preserve surrounding quotes so tokens like 'self' remain quoted
                        parts.append(s)
                    parsed = parts

                # If parsed is an empty list treat it as not set (don't overwrite
                # values from appsettings.json). This prevents empty env vars
                # from wiping JSON-provided CSP values.
                if parsed:
                    try:
                        setattr(ConfigLoader.__appsettings.security, field, parsed)
                        logger.info(f"Applied env override for security.{field}")
                    except Exception:
                        logger.warning(f"Failed to set security.{field} from environment")
                else:
                    logger.debug(f"Skipped empty env override for security.{field}")
        except Exception as e:
            logger.debug(f"Error while applying CSP env overrides: {e}")

        # Determine environment and set production flag
        env = ConfigLoader._getenv_first("FLOUDS_API_ENV") or "Production"
        env_l = str(env).lower()
        ConfigLoader.__appsettings.app.is_production = env_l == "production"

        # Server overrides: use FLOUDS_* env names.
        server_port = ConfigLoader._getenv_first("FLOUDS_PORT")
        if server_port is not None:
            parsed_port = ConfigLoader._parse_int(server_port)
            if parsed_port is not None:
                ConfigLoader.__appsettings.server.port = parsed_port
            else:
                logger.warning(f"Invalid SERVER PORT value: {server_port}; using config value")
        server_host = ConfigLoader._getenv_first("FLOUDS_HOST")
        if server_host:
            ConfigLoader.__appsettings.server.host = server_host

        # OpenAPI public URL (used by docs UI to fetch schema from the correct origin)
        openapi_url = ConfigLoader._getenv_first("FLOUDS_OPENAPI_URL")
        if openapi_url:
            ConfigLoader.__appsettings.server.openapi_url = openapi_url

        # Docs assets configuration (env overrides)
        docs_asset_base = ConfigLoader._getenv_first("FLOUDS_DOCS_ASSET_BASE")
        if docs_asset_base is not None:
            ConfigLoader.__appsettings.server.docs_asset_base = docs_asset_base

        docs_use_proxy = ConfigLoader._getenv_first("FLOUDS_DOCS_USE_PROXY")
        parsed_docs_use_proxy = ConfigLoader._parse_bool(docs_use_proxy)
        if parsed_docs_use_proxy is not None:
            ConfigLoader.__appsettings.server.docs_use_proxy = parsed_docs_use_proxy

        # Debug mode
        debug_val = ConfigLoader._getenv_first("APP_DEBUG_MODE")
        parsed_debug = ConfigLoader._parse_bool(debug_val)
        if parsed_debug is not None:
            ConfigLoader.__appsettings.app.debug = parsed_debug

        # ONNX settings (allow override via env)
        onnx_root = ConfigLoader._getenv_first("FLOUDS_ONNX_ROOT")
        if onnx_root:
            ConfigLoader.__appsettings.onnx.onnx_path = onnx_root
        onnx_cfg = ConfigLoader._getenv_first("FLOUDS_ONNX_CONFIG_FILE")
        if onnx_cfg:
            ConfigLoader.__appsettings.onnx.config_file = onnx_cfg

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

        # Model session provider override
        session_provider = ConfigLoader._getenv_first("FLOUDS_MODEL_SESSION_PROVIDER")
        if session_provider:
            ConfigLoader.__appsettings.server.session_provider = session_provider

        # Rate limiting overrides
        sec_val = ConfigLoader._getenv_first("FLOUDS_RATE_LIMIT_ENABLED")
        parsed_sec = ConfigLoader._parse_bool(sec_val)
        if parsed_sec is not None:
            ConfigLoader.__appsettings.rate_limiting.enabled = parsed_sec

        rpm = ConfigLoader._getenv_first("FLOUDS_RATE_LIMIT_PER_MINUTE")
        parsed_rpm = ConfigLoader._parse_int(rpm)
        if parsed_rpm is not None:
            ConfigLoader.__appsettings.rate_limiting.requests_per_minute = parsed_rpm
        elif rpm is not None:
            logger.warning(f"Invalid rate limit per minute: {rpm}")

        rph = ConfigLoader._getenv_first("FLOUDS_RATE_LIMIT_PER_HOUR")
        parsed_rph = ConfigLoader._parse_int(rph)
        if parsed_rph is not None:
            ConfigLoader.__appsettings.rate_limiting.requests_per_hour = parsed_rph
        elif rph is not None:
            logger.warning(f"Invalid rate limit per hour: {rph}")

        # Misc overrides

        # Cache size overrides for runtime caches (encoder/decoder/models/special tokens)
        enc_cache = ConfigLoader._getenv_first("FLOUDS_ENCODER_CACHE_MAX")
        parsed_enc = ConfigLoader._parse_int(enc_cache)
        if parsed_enc is not None:
            ConfigLoader.__appsettings.cache.encoder_cache_max = parsed_enc
        elif enc_cache is not None:
            logger.warning(f"Invalid encoder cache size: {enc_cache}")

        dec_cache = ConfigLoader._getenv_first("FLOUDS_DECODER_CACHE_MAX")
        parsed_dec = ConfigLoader._parse_int(dec_cache)
        if parsed_dec is not None:
            ConfigLoader.__appsettings.cache.decoder_cache_max = parsed_dec
        elif dec_cache is not None:
            logger.warning(f"Invalid decoder cache size: {dec_cache}")

        models_cache = ConfigLoader._getenv_first("FLOUDS_MODEL_CACHE_MAX")
        parsed_models_cache = ConfigLoader._parse_int(models_cache)
        if parsed_models_cache is not None:
            ConfigLoader.__appsettings.cache.model_cache_max = parsed_models_cache
        elif models_cache is not None:
            logger.warning(f"Invalid model cache max: {models_cache}")

        special_cache = ConfigLoader._getenv_first("FLOUDS_SPECIAL_TOKENS_CACHE_MAX")
        parsed_special = ConfigLoader._parse_int(special_cache)
        if parsed_special is not None:
            ConfigLoader.__appsettings.cache.special_tokens_cache_max = parsed_special
        elif special_cache is not None:
            logger.warning(f"Invalid special tokens cache max: {special_cache}")

        # Generation cache size override
        gen_cache = ConfigLoader._getenv_first("FLOUDS_GENERATION_CACHE_MAX")
        parsed_gen = ConfigLoader._parse_int(gen_cache)
        if parsed_gen is not None:
            ConfigLoader.__appsettings.cache.generation_cache_max = parsed_gen
        elif gen_cache is not None:
            logger.warning(f"Invalid generation cache max: {gen_cache}")

        # Encoder output cache overrides
        enc_out_cache = ConfigLoader._getenv_first("FLOUDS_ENCODER_OUTPUT_CACHE_MAX")
        parsed_enc_out = ConfigLoader._parse_int(enc_out_cache)
        if parsed_enc_out is not None:
            ConfigLoader.__appsettings.cache.encoder_output_cache_max = parsed_enc_out
        elif enc_out_cache is not None:
            logger.warning(f"Invalid encoder output cache max: {enc_out_cache}")

        enc_out_max_bytes = ConfigLoader._getenv_first("FLOUDS_ENCODER_OUTPUT_CACHE_MAX_BYTES")
        parsed_enc_out_bytes = ConfigLoader._parse_int(enc_out_max_bytes)
        if parsed_enc_out_bytes is not None:
            ConfigLoader.__appsettings.cache.encoder_output_cache_max_array_bytes = (
                parsed_enc_out_bytes
            )
        elif enc_out_max_bytes is not None:
            logger.warning(f"Invalid encoder output cache max bytes: {enc_out_max_bytes}")

        max_req = ConfigLoader._getenv_first("FLOUDS_MAX_REQUEST_SIZE")
        parsed_max_req = ConfigLoader._parse_int(max_req)
        if parsed_max_req is not None:
            ConfigLoader.__appsettings.app.max_request_size = parsed_max_req
        elif max_req is not None:
            logger.warning(f"Invalid max request size: {max_req}")

        req_timeout = ConfigLoader._getenv_first("FLOUDS_REQUEST_TIMEOUT")
        parsed_req_timeout = ConfigLoader._parse_int(req_timeout)
        if parsed_req_timeout is not None:
            ConfigLoader.__appsettings.app.request_timeout = parsed_req_timeout
        elif req_timeout is not None:
            logger.warning(f"Invalid request timeout: {req_timeout}")

        # CORS origins are seeded/overridden at application startup (see app/main.py)
        # and therefore are intentionally not read here from the environment.

        # Security enabled flag
        sec_flag = ConfigLoader._getenv_first("FLOUDS_SECURITY_ENABLED")
        parsed_sec_flag = ConfigLoader._parse_bool(sec_flag)
        if parsed_sec_flag is not None:
            ConfigLoader.__appsettings.security.enabled = parsed_sec_flag

        # Background cleanup monitor overrides (enable flag and tuning)
        # Use compact canonical env var prefix for background cleanup
        bg_enabled = ConfigLoader._getenv_first("FLOUDS_BG_CLEANUP_ENABLED")
        parsed_bg_enabled = ConfigLoader._parse_bool(bg_enabled)
        if parsed_bg_enabled is not None:
            ConfigLoader.__appsettings.monitoring.enable_background_cleanup = parsed_bg_enabled

        bg_interval = ConfigLoader._getenv_first("FLOUDS_BG_CLEANUP_INTERVAL_SECONDS")
        parsed_bg_interval = ConfigLoader._parse_int(bg_interval)
        if parsed_bg_interval is not None:
            ConfigLoader.__appsettings.monitoring.background_cleanup_interval_seconds = (
                parsed_bg_interval
            )
        elif bg_interval is not None:
            logger.warning(f"Invalid background cleanup interval: {bg_interval}")

        bg_jitter = ConfigLoader._getenv_first("FLOUDS_BG_CLEANUP_INITIAL_JITTER_SECONDS")
        parsed_bg_jitter = ConfigLoader._parse_int(bg_jitter)
        if parsed_bg_jitter is not None:
            ConfigLoader.__appsettings.monitoring.background_cleanup_initial_jitter_seconds = (
                parsed_bg_jitter
            )
        elif bg_jitter is not None:
            logger.warning(f"Invalid background cleanup initial jitter: {bg_jitter}")

        bg_max_backoff = ConfigLoader._getenv_first("FLOUDS_BG_CLEANUP_MAX_BACKOFF_SECONDS")
        parsed_bg_max_backoff = ConfigLoader._parse_int(bg_max_backoff)
        if parsed_bg_max_backoff is not None:
            ConfigLoader.__appsettings.monitoring.background_cleanup_max_backoff_seconds = (
                parsed_bg_max_backoff
            )
        elif bg_max_backoff is not None:
            logger.warning(f"Invalid background cleanup max backoff: {bg_max_backoff}")

        # Trusted hosts from env
        trusted_hosts = ConfigLoader._getenv_first("FLOUDS_TRUSTED_HOSTS")
        if trusted_hosts:
            ConfigLoader.__appsettings.security.trusted_hosts = [
                h.strip() for h in trusted_hosts.split(",") if h.strip()
            ]

        # Clients DB path override
        clients_db = ConfigLoader._getenv_first("FLOUDS_CLIENTS_DB")
        if clients_db:
            ConfigLoader.__appsettings.security.clients_db_path = clients_db

        # Validate and create critical paths
        ConfigLoader._validate_paths()

        logger.info(f"Loaded AppSettings for environment: {env}")
        return ConfigLoader.__appsettings

    @staticmethod
    def _validate_paths():
        """Validate and create critical directories."""
        settings: AppSettings = cast(AppSettings, ConfigLoader.__appsettings)

        # Only validate ONNX paths in production
        if settings.app.is_production:
            # Validate ONNX root path
            if settings.onnx.onnx_path:
                if not os.path.exists(settings.onnx.onnx_path):
                    logger.error(f"ONNX root path does not exist: {settings.onnx.onnx_path}")
                    sys.exit(1)
                if not os.path.isdir(settings.onnx.onnx_path):
                    logger.error(f"ONNX root path is not a directory: {settings.onnx.onnx_path}")
                    sys.exit(1)
                logger.info(f"Validated ONNX root path: {settings.onnx.onnx_path}")

            # Validate ONNX config file
            if settings.onnx.config_file:
                if not os.path.exists(settings.onnx.config_file):
                    logger.error(f"ONNX config file does not exist: {settings.onnx.config_file}")
                    sys.exit(1)
                if not os.path.isfile(settings.onnx.config_file):
                    logger.error(f"ONNX config file is not a file: {settings.onnx.config_file}")
                    sys.exit(1)
                # Validate that the ONNX config file is valid JSON. A malformed
                # config is a fatal startup error.
                try:
                    with open(settings.onnx.config_file, "r", encoding="utf-8") as cf:
                        json.load(cf)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(
                        "ONNX config file is not valid JSON: %s",
                        sanitize_for_log(str(e)),
                    )
                    sys.exit(1)
                except Exception as e:
                    logger.error(
                        "Error reading ONNX config file %s: %s",
                        sanitize_for_log(settings.onnx.config_file),
                        sanitize_for_log(str(e)),
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
                except OSError as e:
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
            logger.info(f"Validated clients database path: {settings.security.clients_db_path}")

        # Create log directory if specified
        log_path = os.getenv("FLOUDS_LOG_PATH")
        if log_path:
            if not os.path.exists(log_path):
                try:
                    os.makedirs(log_path, exist_ok=True)
                    logger.info(f"Created log directory: {log_path}")
                except OSError as e:
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
        settings: AppSettings = cast(AppSettings, ConfigLoader.__appsettings)
        config_file_name = settings.onnx.config_file

        # Check if cache needs refresh
        if ConfigLoader._should_refresh_cache(config_file_name):
            ConfigLoader._refresh_onnx_cache(config_file_name)

        if ConfigLoader.__onnx_config_cache is None or (
            key not in ConfigLoader.__onnx_config_cache
        ):
            raise MissingConfigError(f"Model config '{key}' not found in onnx_config.json")
        return ConfigLoader.__onnx_config_cache[key]

    @staticmethod
    def _should_refresh_cache(config_file_name: str) -> bool:
        """Check if cache should be refreshed based on file modification time."""
        if ConfigLoader.__onnx_config_cache is None:
            return True

        try:
            current_mtime = os.path.getmtime(config_file_name)
            return ConfigLoader.__config_file_mtime != current_mtime
        except OSError:
            return True

    @staticmethod
    def _refresh_onnx_cache(config_file_name: str) -> None:
        """Refresh the ONNX configuration cache."""
        try:
            raw = ConfigLoader._load_config_data(config_file_name)
            data = cast(Dict[str, Dict[str, Any]], raw)
            # Filter out documentation/metadata keys that start with underscore
            ConfigLoader.__onnx_config_cache = {
                k: OnnxConfig(**v) for k, v in data.items() if not k.startswith("_")
            }
            ConfigLoader.__config_file_mtime = os.path.getmtime(config_file_name)
            logger.debug(
                f"Refreshed ONNX config cache with {len(ConfigLoader.__onnx_config_cache)} models"
            )
        except OSError as e:
            logger.error(f"ONNX config file not accessible: {e}")
            raise MissingConfigError(f"Cannot access ONNX config file: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid ONNX config format: {e}")
            raise InvalidConfigError(f"ONNX config file format error: {e}")
        except Exception as e:
            logger.error(f"Failed to refresh ONNX config cache: {e}")
            raise CacheInvalidationError(f"Cannot refresh config cache: {e}")

    @staticmethod
    def _load_config_data(config_file_name: str, check_env_file: bool = False) -> Dict[str, Any]:
        """
        Loads a config file and merges with environment-specific override if present.
        Performs a deep merge for nested config sections.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, config_file_name)

        logger.debug(f"Loading config from {config_file_name}")

        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> None:
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v

        with open(config_path, "r", encoding="utf-8") as f:
            data = cast(Dict[str, Any], json.load(f))

        # Merge environment-specific config if requested and it exists (deep merge)
        if check_env_file:
            env = os.getenv("FLOUDS_API_ENV", "Production")
            name, ext = os.path.splitext(config_file_name)
            env_file = f"{name}.{env.lower()}{ext}"
            env_path = os.path.join(base_dir, env_file)
            logger.debug(f"Loading config from {env_file}")
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    env_data = cast(Dict[str, Any], json.load(f))
                deep_update(data, env_data)
            except OSError:
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
    def get_cache_stats() -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            "onnx_configs_cached": (
                len(ConfigLoader.__onnx_config_cache) if ConfigLoader.__onnx_config_cache else 0
            ),
            "cache_file_mtime": ConfigLoader.__config_file_mtime,
            "cache_loaded": ConfigLoader.__onnx_config_cache is not None,
        }


# Example usage:
# settings = ConfigLoader.get_app_settings()
# onnx_cfg = ConfigLoader.get_onnx_config()
