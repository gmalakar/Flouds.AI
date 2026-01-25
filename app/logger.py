# =============================================================================
# File: logger.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import inspect
import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional

# Context variables for request-scoped logging fields
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
tenant_code_var: ContextVar[Optional[str]] = ContextVar("tenant_code", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
request_path_var: ContextVar[Optional[str]] = ContextVar("request_path", default=None)
request_method_var: ContextVar[Optional[str]] = ContextVar("request_method", default=None)
request_duration_var: ContextVar[Optional[float]] = ContextVar("request_duration", default=None)


class ContextFilter(logging.Filter):
    """Inject request context into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        record.tenant_code = tenant_code_var.get() or "-"
        record.user_id = user_id_var.get() or "-"
        record.request_path = request_path_var.get() or "-"
        record.request_method = request_method_var.get() or "-"
        record.request_duration = request_duration_var.get()
        return True


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "tenant_code": getattr(record, "tenant_code", "-"),
            "user_id": getattr(record, "user_id", "-"),
        }
        # Request metadata
        path = getattr(record, "request_path", None)
        method = getattr(record, "request_method", None)
        duration = getattr(record, "request_duration", None)
        if path and path != "-":
            base["path"] = path
        if method and method != "-":
            base["method"] = method
        if duration is not None:
            base["duration_ms"] = round(duration * 1000, 2)
        # Include basic source info
        base["module"] = record.module
        base["func"] = record.funcName
        base["line"] = record.lineno
        return json.dumps(base, ensure_ascii=False)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    # Auto-detect caller if name not provided
    if name is None:
        frame = inspect.currentframe()
        caller = frame.f_back if frame and getattr(frame, "f_back", None) else frame
        if caller and getattr(caller, "f_code", None):
            filename = caller.f_code.co_filename
        else:
            filename = __file__
        name = os.path.splitext(os.path.basename(filename))[0]

    # Return logger with caller-specific name
    return _get_or_create_logger(f"flouds.{name}")


# Track configured loggers
_configured_loggers = set()


def _get_or_create_logger(logger_name: str) -> logging.Logger:
    """Get or create logger with specific name."""
    logger = logging.getLogger(logger_name)

    # Only configure if not already configured
    if logger_name in _configured_loggers:
        return logger

    _configured_loggers.add(logger_name)
    is_production = os.getenv("FLOUDS_API_ENV", "Production").lower() == "production"
    # Determine log directory
    if is_production:
        log_dir = os.getenv("FLOUDS_LOG_PATH", "/flouds-ai/logs")
    else:
        # Get parent directory of app folder and create logs folder there
        current_dir = os.path.dirname(os.path.abspath(__file__))  # app folder
        parent_dir = os.path.dirname(current_dir)  # parent of app
        log_dir = os.path.join(parent_dir, "logs")

    # Determine log level
    # Priority: FLOUDS_LOG_LEVEL env -> APP_DEBUG_MODE/dev defaults
    level_env = os.getenv("FLOUDS_LOG_LEVEL")
    if level_env:
        mapped = getattr(logging, level_env.upper(), None)
        level = mapped if isinstance(mapped, int) else logging.INFO
    else:
        if is_production:
            level = logging.DEBUG if os.getenv("APP_DEBUG_MODE", "0") == "1" else logging.INFO
        else:
            level = logging.DEBUG

    # Add date to log file name
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = f"flouds-ai-{date_str}.log"
    max_bytes = int(os.getenv("FLOUDS_LOG_MAX_FILE_SIZE", "10485760"))
    backup_count = int(os.getenv("FLOUDS_LOG_BACKUP_COUNT", "5"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    logger.setLevel(level)

    # Choose formatter: JSON or plain text
    # Default JSON logging only in production; plain text by default in development unless explicitly overridden
    default_json_flag = "1" if is_production else "0"
    use_json = os.getenv("FLOUDS_LOG_JSON", default_json_flag) == "1"
    if use_json:
        formatter: logging.Formatter = JSONFormatter()
    else:
        log_format = os.getenv(
            "FLOUDS_LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        formatter = logging.Formatter(log_format)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.addFilter(ContextFilter())
    logger.addHandler(ch)

    # Rotating file handler
    try:
        fh = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        fh.addFilter(ContextFilter())
        logger.addHandler(fh)
    except OSError as e:
        print(f"Warning: Failed to create log file handler: {e}")
    except (ValueError, TypeError) as e:
        print(f"Warning: Invalid log configuration: {e}")
    except Exception as e:
        print(f"Warning: Unexpected error creating log file handler: {e}")

    return logger


def set_request_context(
    request_id: Optional[str] = None,
    tenant_code: Optional[str] = None,
    user_id: Optional[str] = None,
    request_path: Optional[str] = None,
    request_method: Optional[str] = None,
    request_duration: Optional[float] = None,
) -> None:
    """Set request-scoped logging context variables."""
    if request_id is not None:
        request_id_var.set(request_id)
    if tenant_code is not None:
        tenant_code_var.set(tenant_code)
    if user_id is not None:
        user_id_var.set(user_id)
    if request_path is not None:
        request_path_var.set(request_path)
    if request_method is not None:
        request_method_var.set(request_method)
    if request_duration is not None:
        request_duration_var.set(request_duration)
