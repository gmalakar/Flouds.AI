# =============================================================================
# File: log_context.py
# Date: 2026-01-17
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: log_context.py
# Date: 2026-01-17
# =============================================================================

import json
import time
import uuid
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.logger import get_logger, set_request_context


def _sanitize_body(body: bytes, max_length: int = 200) -> str:
    """Safely sanitize request body for logging.

    Args:
        body: Raw request body bytes
        max_length: Maximum characters to include

    Returns:
        Sanitized string representation
    """
    if not body or len(body) == 0:
        return ""

    try:
        # Try to parse as JSON
        parsed = json.loads(body)
        # Remove sensitive keys
        sensitive_keys = {
            "password",
            "token",
            "secret",
            "api_key",
            "apikey",
            "authorization",
        }
        if isinstance(parsed, dict):
            sanitized = {
                k: "***REDACTED***" if k.lower() in sensitive_keys else v for k, v in parsed.items()
            }
            snippet = json.dumps(sanitized)[:max_length]
            return snippet + "..." if len(snippet) >= max_length else snippet
        else:
            snippet = json.dumps(parsed)[:max_length]
            return snippet + "..." if len(snippet) >= max_length else snippet
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Non-JSON body; return safe snippet
        try:
            decoded = body.decode("utf-8", errors="replace")[:max_length]
            return decoded + "..." if len(decoded) >= max_length else decoded
        except Exception:
            return f"<binary {len(body)} bytes>"


class LogContextMiddleware(BaseHTTPMiddleware):
    """Populate logging context variables per request and set X-Request-ID.

    Tracks request timing, endpoint metadata, and optionally logs safe payload snippets.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Any:
        # Start timing
        start_time = time.perf_counter()

        # Extract context from headers
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        tenant_code = request.headers.get("X-Tenant-Code")
        user_id = request.headers.get("X-User-ID")

        # Capture endpoint metadata
        path = str(request.url.path)
        method = request.method

        # Set initial logging context (without duration)
        set_request_context(
            request_id=request_id,
            tenant_code=tenant_code,
            user_id=user_id,
            request_path=path,
            request_method=method,
        )

        # Continue request
        response = await call_next(request)

        # Calculate duration
        duration = time.perf_counter() - start_time

        # Update context with duration for final log
        set_request_context(request_duration=duration)

        # Echo back request id for correlation
        try:
            response.headers["X-Request-ID"] = request_id
        except Exception:
            pass

        # Log request lifecycle with metadata
        log = get_logger("request")
        log.info(f"{method} {path} -> {response.status_code} " f"[{duration*1000:.2f}ms]")

        return response
