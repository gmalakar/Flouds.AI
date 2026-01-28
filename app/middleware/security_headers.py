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

from typing import Any, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config.appsettings import SecurityConfig
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
            "accelerometer=(), autoplay=(), "
            "camera=(), encrypted-media=(), fullscreen=(), geolocation=(), "
            "gyroscope=(), magnetometer=(), microphone=(), payment=(), "
            "usb=()"
        ),
    }

    @staticmethod
    def build_csp(security_config: SecurityConfig, is_production: bool) -> str:
        """Build Content-Security-Policy from SecurityConfig."""
        # Coerce None to sensible defaults to be defensive when config loader
        # hasn't provided CSP arrays. Use a minimal safe default.
        script_src_list = (
            list(security_config.csp_script_src or [])
            if getattr(security_config, "csp_script_src", None) is not None
            else ["'self'"]
        )
        style_src_list = (
            list(security_config.csp_style_src or [])
            if getattr(security_config, "csp_style_src", None) is not None
            else ["'self'", "'unsafe-inline'"]
        )
        img_src_list = (
            list(security_config.csp_img_src or [])
            if getattr(security_config, "csp_img_src", None) is not None
            else ["'self'", "data:", "https:"]
        )

        def _normalize_token(tok: str) -> str:
            t = str(tok).strip()
            # strip surrounding quotes if present
            if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
                t = t[1:-1].strip()
            # ensure special keywords are single-quoted
            keywords = {"self", "unsafe-inline", "unsafe-eval", "none"}
            if t in keywords:
                return f"'{t}'"
            return t

        script_src = " ".join([_normalize_token(s) for s in script_src_list])
        style_src = " ".join([_normalize_token(s) for s in style_src_list])
        img_src = " ".join([_normalize_token(s) for s in img_src_list])

        # Allow websockets in development
        connect_src_list = (
            list(security_config.csp_connect_src or [])
            if getattr(security_config, "csp_connect_src", None) is not None
            else ["'self'"]
        )
        if not is_production:
            connect_src_list.extend(["localhost:*", "ws:"])
        connect_src_str = " ".join([_normalize_token(s) for s in connect_src_list])

        # Font sources (allow Google fonts via config)
        font_src_list = (
            list(security_config.csp_font_src or [])
            if getattr(security_config, "csp_font_src", None) is not None
            else ["'self'"]
        )
        font_src = " ".join([_normalize_token(s) for s in font_src_list])

        # Worker sources (for blob workers)
        worker_src_list = (
            list(security_config.csp_worker_src or [])
            if getattr(security_config, "csp_worker_src", None) is not None
            else ["'self'", "blob:"]
        )
        worker_src = " ".join([_normalize_token(s) for s in worker_src_list])

        return (
            f"default-src 'self'; "
            f"script-src {script_src}; "
            f"style-src {style_src}; "
            f"img-src {img_src}; "
            f"font-src {font_src}; "
            f"connect-src {connect_src_str}; "
            f"worker-src {worker_src}; "
            f"frame-ancestors 'none'; "
            f"base-uri 'self'; "
            f"form-action 'self'"
        )

    def __init__(
        self,
        app: Any,
        is_production: bool = True,
        security_config: Optional[SecurityConfig] = None,
    ):
        super().__init__(app)
        self.is_production = is_production
        self.security_config = security_config or SecurityConfig()

        # Build headers dictionary
        self.headers_to_add = dict(self.SECURITY_HEADERS)

        # Add HSTS in production if enabled
        if is_production and self.security_config.enable_hsts:
            self.headers_to_add["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Build CSP from config
        self.headers_to_add["Content-Security-Policy"] = self.build_csp(
            self.security_config, is_production
        )

        logger.info(
            f"SecurityHeadersMiddleware initialized (production={is_production}, "
            f"headers={len(self.headers_to_add)}, hsts={self.security_config.enable_hsts})"
        )

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Any:
        """Add security headers to response."""
        response = await call_next(request)

        # Add all configured security headers
        for header_name, header_value in self.headers_to_add.items():
            response.headers[header_name] = header_value

        return response
