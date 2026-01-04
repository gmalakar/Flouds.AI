# =============================================================================
# File: path_security.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Middleware for path security validation."""

import re
from typing import Any, Awaitable, Callable, cast

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.logger import get_logger
from app.utils.log_sanitizer import sanitize_for_log

logger = get_logger("path_security")

# Patterns that indicate potential path traversal attempts
SUSPICIOUS_PATTERNS = [
    r"\.\./",  # Parent directory traversal
    r"\.\.\\",  # Windows parent directory traversal
    r"~[/\\]",  # Home directory access
    r"\$[A-Z_]+",  # Environment variables
    r"%[A-Z_]+%",  # Windows environment variables
    r"\\\\[^\\]+\\",  # UNC paths
    r"/etc/",  # Unix system directories
    r"/proc/",  # Unix process directories
    r"/sys/",  # Unix system directories
    r"C:\\Windows\\",  # Windows system directory
    r"C:\\Program Files",  # Windows program directory
]

COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in SUSPICIOUS_PATTERNS
]


class PathSecurityMiddleware(BaseHTTPMiddleware):
    """Middleware to detect and block path traversal attempts in requests."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        logger.info("Path security middleware initialized")

    def _scan_value(self, value: Any, path: str = "") -> bool:
        """Recursively scan values for suspicious patterns."""
        if isinstance(value, str):
            for pattern in COMPILED_PATTERNS:
                if pattern.search(value):
                    logger.warning(
                        "Suspicious path pattern detected in %s: %s",
                        sanitize_for_log(path),
                        sanitize_for_log(value[:100]),
                    )
                    return True
        elif isinstance(value, dict):
            d = cast(dict[str, Any], value)
            for key, val in d.items():
                if self._scan_value(val, f"{path}.{key}"):
                    return True
        elif isinstance(value, list):
            lst = cast(list[Any], value)
            for i, val in enumerate(lst):
                if self._scan_value(val, f"{path}[{i}]"):
                    return True
        return False

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request and scan for path traversal attempts."""

        # Skip health checks and static content
        if request.url.path.startswith(
            ("/api/v1/health", "/docs", "/redoc", "/openapi.json")
        ):
            return await call_next(request)

        try:
            # Scan URL path
            if self._scan_value(request.url.path, "url.path"):
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "Suspicious path detected in URL",
                        "error_code": "PATH_TRAVERSAL_DETECTED",
                    },
                )

            # Scan query parameters
            for key, value in request.query_params.items():
                if self._scan_value(value, f"query.{key}"):
                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "message": "Suspicious path detected in query parameters",
                            "error_code": "PATH_TRAVERSAL_DETECTED",
                        },
                    )

            # Skip body scanning to avoid middleware chain issues
            # Body content will be validated at the application level

            return await call_next(request)

        except Exception as e:
            logger.error("Path security middleware error: %s", str(e))
            return await call_next(request)  # Continue on middleware errors
