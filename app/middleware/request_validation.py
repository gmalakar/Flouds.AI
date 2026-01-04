# =============================================================================
# File: request_validation.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import time
from typing import Awaitable, Callable

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.utils.log_sanitizer import sanitize_for_log

logger = get_logger("request_validation")


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation, size limits, and timeout handling."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.max_request_size = APP_SETTINGS.app.max_request_size
        self.request_timeout = APP_SETTINGS.app.request_timeout

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request with validation and timeout."""

        # Skip validation for health checks and docs
        if request.url.path.startswith(
            (
                "/api/v1/health",
                "/api/v1/docs",
                "/api/v1/redoc",
                "/api/v1/openapi.json",
                "/docs",
                "/redoc",
                "/openapi.json",
            )
        ):
            return await call_next(request)

        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning(
                "Request size %s exceeds limit %d",
                sanitize_for_log(content_length),
                self.max_request_size,
            )
            raise HTTPException(
                status_code=413,
                detail={
                    "success": False,
                    "error_code": "REQUEST_TOO_LARGE",
                    "message": f"Request size exceeds maximum allowed size of {self.max_request_size} bytes",
                    "max_size_bytes": self.max_request_size,
                },
            )

        # Add request start time for timeout tracking
        start_time = time.time()
        request.state.start_time = start_time

        try:
            # Process request
            response = await call_next(request)

            # Add processing time header
            processing_time = time.time() - start_time
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"

            # Log slow requests
            if processing_time > 5.0:  # Log requests taking more than 5 seconds
                logger.warning(
                    "Slow request: %s %s took %.3fs",
                    sanitize_for_log(request.method),
                    sanitize_for_log(request.url.path),
                    processing_time,
                )

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Request failed after {processing_time:.3f}s: {str(e)}")
            raise
