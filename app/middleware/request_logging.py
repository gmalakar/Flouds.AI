# =============================================================================
# File: request_logging.py
# Date: 2026-01-30
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: request_logging.py
# Date: 2026-01-30
# =============================================================================

import json
import time
import uuid
from typing import Any, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.utils.log_sanitizer import sanitize_for_log

logger = get_logger("request_logging")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging incoming requests and outgoing responses.

    Logs request metadata, sanitized request bodies, and response status/duration.
    """

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Any:
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        client_ip = sanitize_for_log(request.client.host if request.client else "unknown")
        method = sanitize_for_log(request.method)
        url = sanitize_for_log(str(request.url))
        user_agent = sanitize_for_log(request.headers.get("user-agent", ""))

        logger.info(f"Request[{request_id}]: {method} {url} from {client_ip} UA: {user_agent}")

        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    body_size = len(body)
                    cap = APP_SETTINGS.app.max_request_size
                    if body_size > cap:
                        logger.warning(
                            "Request[%s] body size %d exceeds logging cap of %d; truncating for logs",
                            request_id,
                            body_size,
                            cap,
                        )
                        snippet = body[:cap].decode("utf-8", errors="replace")
                        logger.debug(
                            "Request[%s] body (truncated): %s",
                            request_id,
                            sanitize_for_log(snippet),
                        )
                    else:
                        try:
                            json_body = json.loads(body)
                            sanitized_body = self._sanitize_request_body(json_body)
                            logger.debug(
                                "Request[%s] body: %s", request_id, json.dumps(sanitized_body)
                            )
                        except json.JSONDecodeError:
                            snippet = body[:500].decode("utf-8", errors="replace")
                            logger.debug(
                                "Request[%s] body (non-JSON): %s",
                                request_id,
                                sanitize_for_log(snippet),
                            )

                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}

                request._receive = receive
            except Exception as e:
                logger.warning(f"Failed to log request body: {e}")

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        duration = time.time() - start_time
        status_code = response.status_code
        logger.info(f"Response[{request_id}]: {status_code} for {method} {url} in {duration:.3f}s")
        return response

    def _sanitize_request_body(self, data: Any) -> Any:
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                key_lower = key.lower()
                if key_lower in ["password", "secret", "token", "key", "auth"]:
                    sanitized[key] = "[REDACTED]"
                elif key == "vector" and isinstance(value, list):
                    sanitized[key] = f"[vector with {len(value)} dimensions]"
                elif key == "data" and isinstance(value, list):
                    sanitized[key] = f"[array with {len(value)} items]"
                else:
                    sanitized[key] = sanitize_for_log(str(value))
            return sanitized
        elif isinstance(data, list):
            return f"[array with {len(data)} items]"
        else:
            return sanitize_for_log(str(data))
