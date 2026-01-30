# =============================================================================
# File: request_size_limit.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Middleware to enforce maximum request size limits.

Prevents denial-of-service attacks from extremely large requests that could
exhaust server memory or cause processing delays.
"""

from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.app_init import APP_SETTINGS
from app.logger import get_logger

logger = get_logger(__name__)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Enforce maximum request size from configuration.

    Attributes:
        # Maximum request size in bytes (from APP_SETTINGS)
    """

    def __init__(self, app: Any):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Any:
        """Check request size before processing."""
        # Get Content-Length header
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                content_length_int = int(content_length)
                effective_max = APP_SETTINGS.app.max_request_size
                if effective_max is not None and content_length_int > effective_max:
                    logger.warning(
                        f"Request size {content_length_int} bytes exceeds limit {effective_max} bytes "
                        f"from {request.client.host if request.client else 'unknown'}"
                    )
                    return JSONResponse(
                        status_code=413,  # Payload Too Large
                        content={
                            "success": False,
                            "message": "Request payload too large",
                            "error_code": "PAYLOAD_TOO_LARGE",
                            "detail": f"Maximum request size is {effective_max} bytes",
                            "max_size": effective_max,
                        },
                    )
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid content-length header: {e}")

        # Request size is acceptable or not provided
        return await call_next(request)
