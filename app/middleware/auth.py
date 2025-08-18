# =============================================================================
# File: auth.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from fastapi import Request, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.app_init import APP_SETTINGS
from app.exceptions import InvalidTokenError, UnauthorizedError
from app.logger import get_logger
from app.models.base_response import BaseResponse
from app.utils.key_manager import key_manager
from app.utils.log_sanitizer import sanitize_for_log
from app.utils.performance_tracker import perf_tracker

logger = get_logger("auth")

security = HTTPBearer()


class AuthMiddleware(BaseHTTPMiddleware):
    """API Key authentication middleware."""

    def __init__(self, app):
        super().__init__(app)
        self.enabled = APP_SETTINGS.security.enabled

        # Cache public endpoints for faster lookup
        self.public_endpoints = frozenset(
            [
                "/",
                "/api/v1",
                "/api/v1/health",
                "/api/v1/health/live",
                "/api/v1/health/ready",
                "/api/v1/docs",
                "/api/v1/redoc",
                "/api/v1/openapi.json",
            ]
        )

        # Log security status on startup
        if self.enabled:
            valid_keys = key_manager.get_all_tokens()
            if valid_keys:
                logger.info(
                    f"API authentication enabled with {len(valid_keys)} client(s)"
                )
            else:
                logger.warning("API authentication enabled but no clients configured")
        else:
            logger.info("API authentication disabled")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with API key authentication."""

        # Skip auth for public endpoints (optimized lookup)
        path = request.url.path
        if path in self.public_endpoints or path.startswith("/api/v1/health/"):
            return await call_next(request)

        # Skip if auth is disabled
        if not self.enabled:
            return await call_next(request)

        # Check if any API keys are configured (use key manager directly)
        valid_keys = key_manager.get_all_tokens()
        if not valid_keys:
            logger.error("Authentication enabled but no API keys configured")
            error_response = BaseResponse(
                success=False, message="Authentication misconfigured", model="auth"
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.model_dump(),
            )

        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise UnauthorizedError("Missing Authorization header")

        # Validate Bearer token format and extract token in one step
        if not auth_header.startswith("Bearer ") or len(auth_header) <= 7:
            raise InvalidTokenError(
                "Invalid Authorization format. Use 'Bearer <token>'"
            )

        token = auth_header[7:].strip()  # Remove "Bearer " prefix and trim

        # Validate token is not empty
        if not token:
            raise InvalidTokenError("Empty authorization token")

        # Authenticate client using optimized key manager with performance tracking
        with perf_tracker.track("auth_client_lookup"):
            client = key_manager.authenticate_client(token)
        if client:
            request.state.client_id = client.client_id
            request.state.client_type = client.client_type
            logger.debug(
                "Authenticated client: %s (%s)",
                sanitize_for_log(client.client_id),
                sanitize_for_log(client.client_type),
            )
        else:
            raise InvalidTokenError("Invalid token")

        return await call_next(request)
