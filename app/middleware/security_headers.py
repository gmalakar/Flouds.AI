# =============================================================================
# File: security_headers.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Middleware to add security headers to all HTTP responses.

Implements OWASP recommended security headers to protect against common
web vulnerabilities including XSS, clickjacking, MIME sniffing, etc.

Reference: https://owasp.org/www-project-secure-headers/
"""

from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.logger import get_logger

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all HTTP responses.

    Headers added:
    - X-Content-Type-Options: Prevents MIME type sniffing
    - X-Frame-Options: Prevents clickjacking
    - X-XSS-Protection: Legacy XSS protection (backup)
    - Strict-Transport-Security: Forces HTTPS
    - Content-Security-Policy: Controls resource loading
    - Referrer-Policy: Controls referrer information
    - Permissions-Policy: Controls browser features
    """

    # Security headers to add to all responses
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": (
            "accelerometer=(), ambient-light-sensor=(), autoplay=(), "
            "camera=(), encrypted-media=(), fullscreen=(), geolocation=(), "
            "gyroscope=(), magnetometer=(), microphone=(), payment=(), "
            "usb=()"
        ),
    }

    # Production-only headers (stricter)
    PRODUCTION_HEADERS = {
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        ),
    }

    # Development headers (less strict for debugging)
    DEV_HEADERS = {
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self' localhost:* ws:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        ),
    }

    def __init__(self, app: Any, is_production: bool = True):
        super().__init__(app)
        self.is_production = is_production
        self.headers_to_add = dict(self.SECURITY_HEADERS)
        if is_production:
            self.headers_to_add.update(self.PRODUCTION_HEADERS)
        else:
            self.headers_to_add.update(self.DEV_HEADERS)
        logger.info(
            f"SecurityHeadersMiddleware initialized (production={is_production}, "
            f"headers={len(self.headers_to_add)})"
        )

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Any:
        """Add security headers to response."""
        response = await call_next(request)

        # Add all configured security headers
        for header_name, header_value in self.headers_to_add.items():
            response.headers[header_name] = header_value

        return response
