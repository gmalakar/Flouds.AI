# =============================================================================
# File: request_validation.py
# Date: 2026-01-30
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: request_validation.py
# Canonical RequestValidationMiddleware for Flouds.Py
# =============================================================================

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.app_init import APP_SETTINGS
from app.logger import get_logger, set_request_context
from app.utils.log_sanitizer import sanitize_for_log

logger = get_logger("request_validation")


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation, size limits, and timeout handling.

    Legacy notes:
    - Flouds.Py historically preferred `APP_SETTINGS`; we preserve that preference
      but fall back to `ConfigLoader` and environment vars.
    - Skip validation for health/docs endpoints to preserve legacy behavior.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        # Middleware will read values directly from `APP_SETTINGS.app` at request time.

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Skip validation for health/docs endpoints (legacy behavior)
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

        start_time = time.perf_counter()

        # Use configured values from APP_SETTINGS (no runtime env overrides)
        max_size = APP_SETTINGS.app.max_request_size
        timeout = APP_SETTINGS.app.request_timeout

        # Request context and IDs
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        tenant_code = request.headers.get("X-Tenant-Code")
        user_id = request.headers.get("X-User-ID")
        path = str(request.url.path)
        method = request.method

        set_request_context(
            request_id=request_id,
            tenant_code=tenant_code,
            user_id=user_id,
            request_path=path,
            request_method=method,
        )

        # Early Content-Length short-circuit
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                cl = int(content_length)
                if max_size is not None and cl > int(max_size):
                    logger.warning(
                        "Request Content-Length %s exceeds limit %d",
                        sanitize_for_log(content_length),
                        max_size,
                    )
                    detail = {
                        "success": False,
                        "error_code": "REQUEST_TOO_LARGE",
                        "message": f"Request size exceeds maximum allowed size of {max_size} bytes",
                        "max_size_bytes": max_size,
                    }
                    resp = JSONResponse(content=detail, status_code=413)
                    resp.headers["X-Request-ID"] = request_id
                    return resp
            except Exception:
                pass

        # For mutating requests, validate Content-Type, read JSON bodies and
        # enforce max size. Allow multipart/form-data and
        # application/x-www-form-urlencoded for file/form uploads without
        # reading the entire body here (downstream handlers will parse it).
        if method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            ct = content_type.lower() if content_type else ""

            # Accept JSON, multipart form-data, and urlencoded forms
            if ct.startswith("application/json"):
                try:
                    body = await request.body()
                    if body and (max_size is not None) and len(body) > int(max_size):
                        logger.warning(
                            "Request body size %d exceeds limit %d; returning 413",
                            len(body),
                            max_size,
                        )
                        resp = JSONResponse(
                            status_code=413,
                            content={
                                "error": "Payload Too Large",
                                "message": f"Request body {len(body)} bytes exceeds allowed limit {max_size} bytes",
                            },
                        )
                        resp.headers["X-Request-ID"] = request_id
                        return resp

                    async def _receive():
                        return {"type": "http.request", "body": body, "more_body": False}

                    request._receive = _receive
                except Exception:
                    # If body cannot be read, let downstream handle it
                    pass
            elif ct.startswith("multipart/form-data") or ct.startswith(
                "application/x-www-form-urlencoded"
            ):
                # Allow file/form uploads: do not eagerly read body here to avoid
                # buffering large uploads. Downstream handlers (FastAPI)
                # will parse the form/multipart stream.
                pass
            else:
                logger.warning("Invalid content type: %s", sanitize_for_log(content_type))
                return JSONResponse(status_code=415, content={"detail": "Unsupported media type"})

        # Execute downstream with timeout
        request.state.start_time = time.time()
        try:
            try:
                response = await asyncio.wait_for(call_next(request), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "Request timed out after %ds: %s %s",
                    timeout,
                    sanitize_for_log(method),
                    sanitize_for_log(path),
                )
                detail = {
                    "success": False,
                    "error_code": "REQUEST_TIMEOUT",
                    "message": f"Request processing exceeded timeout of {timeout} seconds",
                }
                resp = JSONResponse(content=detail, status_code=504)
                resp.headers["X-Request-ID"] = request_id
                return resp

            processing_time = time.perf_counter() - start_time
            try:
                response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
            except Exception:
                pass

            if processing_time > 5.0:
                logger.warning(
                    "Slow request: %s %s took %.3fs",
                    sanitize_for_log(method),
                    sanitize_for_log(path),
                    processing_time,
                )

            try:
                response.headers["X-Request-ID"] = request_id
            except Exception:
                pass

            set_request_context(request_duration=processing_time)

            logger.info(
                "%s %s -> %s [%.2fms]",
                method,
                path,
                getattr(response, "status_code", "unknown"),
                processing_time * 1000,
            )
            return response

        except Exception:
            processing_time = time.perf_counter() - start_time
            logger.exception("Request failed after %.3fs", processing_time)
            raise
