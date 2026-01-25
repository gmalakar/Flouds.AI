# =============================================================================
# File: request_size_limit.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Middleware to enforce maximum request size limits.

Prevents denial-of-service attacks from extremely large requests that could
exhaust server memory or cause processing delays.
"""

from typing import Any, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.app_init import APP_SETTINGS
from app.logger import get_logger

logger = get_logger(__name__)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Enforce maximum request size from configuration.

    Attributes:
        max_size: Maximum request size in bytes (from APP_SETTINGS)
    """

    def __init__(self, app: Any, max_size: Optional[int] = None):
        super().__init__(app)
        # Use provided max_size or fall back to app-level settings
        if max_size is None:
            resolved_max = APP_SETTINGS.app.max_request_size
        else:
            resolved_max = max_size
        self.max_size: int = int(resolved_max)
        logger.info(
            f"RequestSizeLimitMiddleware configured: max_size={self.max_size} bytes"
        )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Any]
    ) -> Any:
        """Check request size before processing."""
        # Get Content-Length header
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                content_length_int = int(content_length)
                if content_length_int > self.max_size:
                    logger.warning(
                        f"Request size {content_length_int} bytes exceeds limit {self.max_size} bytes "
                        f"from {request.client.host if request.client else 'unknown'}"
                    )
                    return JSONResponse(
                        status_code=413,  # Payload Too Large
                        content={
                            "success": False,
                            "message": "Request payload too large",
                            "error_code": "PAYLOAD_TOO_LARGE",
                            "detail": f"Maximum request size is {self.max_size} bytes",
                            "max_size": self.max_size,
                        },
                    )
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid content-length header: {e}")

        # Request size is acceptable or not provided
        return await call_next(request)
