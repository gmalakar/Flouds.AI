# =============================================================================
# File: error_handler.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Centralized error handling utilities."""

import functools
import traceback
from typing import Any, Callable, Dict

from app.exceptions import (
    AuthenticationException,
    CacheException,
    ConfigurationException,
    DatabaseException,
    EncryptionException,
    FloudsBaseException,
    HealthCheckException,
    ModelException,
    RateLimitException,
    ResourceException,
    TimeoutException,
    ValidationException,
)
from app.logger import get_logger
from app.utils.log_sanitizer import sanitize_for_log

logger = get_logger("error_handler")


class ErrorHandler:
    """Centralized error handling and response formatting."""

    ERROR_MAPPINGS = {
        FileNotFoundError: ("Model files not accessible", "FILE_NOT_FOUND"),
        OSError: ("System resource error", "SYSTEM_ERROR"),
        ValueError: ("Invalid parameter value", "INVALID_VALUE"),
        KeyError: ("Missing required parameter", "MISSING_PARAMETER"),
        TypeError: ("Invalid parameter type", "INVALID_TYPE"),
        TimeoutError: ("Operation timed out", "TIMEOUT"),
        MemoryError: ("Insufficient memory", "MEMORY_ERROR"),
        RuntimeError: ("Runtime error occurred", "RUNTIME_ERROR"),
    }

    @staticmethod
    def handle_exception(
        exc: Exception, context: str = "operation", include_traceback: bool = False
    ) -> Dict[str, Any]:
        """Handle exception and return standardized error response."""

        if isinstance(exc, FloudsBaseException):
            error_code = exc.error_code
            message = exc.message
            log_level = "warning"
        else:
            message, error_code = ErrorHandler.ERROR_MAPPINGS.get(
                type(exc), (str(exc), "UNKNOWN_ERROR")
            )
            log_level = "error"

        # Log the error with sanitization
        log_message = f"{context} failed: {sanitize_for_log(message)}"
        if log_level == "error":
            logger.error(log_message)
            if include_traceback:
                logger.error("Traceback: %s", sanitize_for_log(traceback.format_exc()))
        else:
            logger.warning(log_message)

        return {
            "success": False,
            "message": message,
            "error_code": error_code,
            "context": context,
        }

    @staticmethod
    def get_http_status(exc: Exception) -> int:
        """Get appropriate HTTP status code for exception."""

        if isinstance(exc, ValidationException):
            return 400
        elif isinstance(exc, AuthenticationException):
            return 401
        elif isinstance(exc, RateLimitException):
            return 429
        elif isinstance(exc, (ModelException, ConfigurationException)):
            return 503
        elif isinstance(exc, TimeoutException):
            return 504
        elif isinstance(exc, ResourceException):
            return 507
        elif isinstance(exc, (DatabaseException, EncryptionException)):
            return 500
        elif isinstance(exc, CacheException):
            return 503
        elif isinstance(exc, HealthCheckException):
            return 503
        else:
            return 500


def handle_errors(
    context: str = "operation", reraise: bool = False, include_traceback: bool = False
) -> Callable:
    """Decorator for automatic error handling."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_response = ErrorHandler.handle_exception(e, context, include_traceback)

                if reraise:
                    raise

                # Return error response in expected format
                if hasattr(func, "__annotations__"):
                    return_type = func.__annotations__.get("return")
                    if return_type and hasattr(return_type, "__name__"):
                        if "Response" in return_type.__name__:
                            # Create response object with error
                            return return_type(
                                success=False,
                                message=error_response["message"],
                                results=[],
                                time_taken=0.0,
                            )

                return error_response

        return wrapper

    return decorator


def handle_async_errors(
    context: str = "async_operation",
    reraise: bool = False,
    include_traceback: bool = False,
) -> Callable:
    """Decorator for automatic async error handling."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_response = ErrorHandler.handle_exception(e, context, include_traceback)

                if reraise:
                    raise

                # Return error response in expected format
                if hasattr(func, "__annotations__"):
                    return_type = func.__annotations__.get("return")
                    if return_type and hasattr(return_type, "__name__"):
                        if "Response" in return_type.__name__:
                            # Create response object with error
                            return return_type(
                                success=False,
                                message=error_response["message"],
                                results=[],
                                time_taken=0.0,
                            )

                return error_response

        return wrapper

    return decorator
