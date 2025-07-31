# =============================================================================
# File: auth.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from fastapi import Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.models.base_response import BaseResponse
from app.utils.key_manager import key_manager

logger = get_logger("auth")

security = HTTPBearer()


class AuthMiddleware(BaseHTTPMiddleware):
    """API Key authentication middleware."""

    def __init__(self, app):
        super().__init__(app)
        self.enabled = APP_SETTINGS.security.enabled

        # Get tokens from key manager
        self.valid_keys = set(key_manager.get_all_tokens())

        # Log security status on startup
        if self.enabled:
            if self.valid_keys:
                logger.info(
                    f"API authentication enabled with {len(self.valid_keys)} client(s)"
                )
            else:
                logger.warning("API authentication enabled but no clients configured")
        else:
            logger.info("API authentication disabled")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with API key authentication."""

        # Define public endpoints that don't require authentication
        public_endpoints = (
            "/",
            "/api/v1",
            "/api/v1/health",
            "/api/v1/health/live",
            "/api/v1/health/ready",
            "/api/v1/docs",
            "/api/v1/redoc",
            "/api/v1/openapi.json",
        )

        # Skip auth for public endpoints
        if request.url.path in public_endpoints or request.url.path.startswith(
            "/api/v1/health/"
        ):
            return await call_next(request)

        # Skip if auth is disabled
        if not self.enabled:
            return await call_next(request)

        # Check if any API keys are configured
        if not self.valid_keys:
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
            logger.warning(f"Missing Authorization header for {request.url.path}")
            error_response = BaseResponse(
                success=False, message="Missing Authorization header", model="auth"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=error_response.model_dump(),
            )

        # Validate Bearer token
        if not auth_header.startswith("Bearer "):
            logger.warning(f"Invalid Authorization format for {request.url.path}")
            error_response = BaseResponse(
                success=False,
                message="Invalid Authorization format. Use 'Bearer <token>'",
                model="auth",
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=error_response.model_dump(),
            )

        token = auth_header[7:].strip()  # Remove "Bearer " prefix and trim

        # Validate token is not empty
        if not token:
            logger.warning(f"Empty token provided for {request.url.path}")
            error_response = BaseResponse(
                success=False, message="Empty authorization token", model="auth"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=error_response.model_dump(),
            )

        # Authenticate client using client_id|client_secret format
        client = key_manager.authenticate_client(token)
        if client:
            request.state.client_id = client.client_id
            request.state.client_type = client.client_type
            logger.debug(
                f"Authenticated client: {client.client_id} ({client.client_type})"
            )
        else:
            logger.warning(f"Invalid token attempt for {request.url.path}")
            error_response = BaseResponse(
                success=False, message="Invalid token", model="auth"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=error_response.model_dump(),
            )

        return await call_next(request)
