# =============================================================================
# File: auth.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.models.base_response import BaseResponse
from app.utils.key_manager import key_manager
from app.utils.performance_tracker import perf_tracker

logger = get_logger("auth")

router = APIRouter()


# Security scheme for dependency injection
class FloudsHTTPBearer(HTTPBearer):
    """Custom HTTPBearer security scheme for Flouds AI."""

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials:
        """Validate and return credentials."""
        # Skip if auth is disabled
        if not APP_SETTINGS.security.enabled:
            return HTTPAuthorizationCredentials(scheme="bearer", credentials="disabled")

        credentials = await super().__call__(request)

        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
            )

        # Validate token
        client = key_manager.authenticate_client(credentials.credentials)
        if not client:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

        # Store client info in request state
        request.state.client_id = client.client_id
        request.state.client_type = client.client_type

        return credentials


# Global security instance
security = FloudsHTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """API Key authentication middleware."""

    def __init__(self, app):
        super().__init__(app)
        self.enabled = APP_SETTINGS.security.enabled

        # Cache valid keys count at startup to avoid repeated calls
        self._keys_configured = (
            bool(key_manager.get_all_tokens()) if self.enabled else True
        )

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
                "/favicon.ico",
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

        # Early exit for disabled auth
        if not self.enabled:
            return await call_next(request)

        # Skip auth for public endpoints (optimized lookup)
        path = request.url.path
        if path in self.public_endpoints or path.startswith("/api/v1/health/"):
            return await call_next(request)

        # Check if keys are configured (cached check)
        if not self._keys_configured:
            logger.error("Authentication enabled but no API keys configured")
            error_response = BaseResponse(
                success=False, message="Authentication misconfigured", model="auth"
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.model_dump(),
            )

        # Extract and validate token from header or query parameter (dev only)
        auth_header = request.headers.get("Authorization")
        token = None

        if auth_header and auth_header.startswith("Bearer ") and len(auth_header) > 7:
            token = auth_header[7:].strip()
        elif not APP_SETTINGS.app.is_production:
            # Check for token in query parameter only in development
            token = request.query_params.get("token")

        if not token:
            error_response = BaseResponse(
                success=False,
                message="Missing Authorization header"
                + (" or token parameter" if not APP_SETTINGS.app.is_production else ""),
                model="auth",
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=error_response.model_dump(),
            )

        # Authenticate client with performance tracking
        with perf_tracker.track("auth_client_lookup"):
            client = key_manager.authenticate_client(token)

        if not client:
            error_response = BaseResponse(
                success=False, message="Invalid token", model="auth"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=error_response.model_dump(),
            )

        # Store client info in request state
        request.state.client_id = client.client_id
        request.state.client_type = client.client_type

        return await call_next(request)


@router.get("/secure-endpoint")
def secure_endpoint(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Example secured endpoint using dependency injection."""
    return {
        "message": "Secured access granted",
        "client_authenticated": credentials.scheme == "bearer"
        and credentials.credentials != "disabled",
    }


# Dependency function for easy reuse
async def get_current_client(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Get current authenticated client ID."""
    if not APP_SETTINGS.security.enabled:
        return "anonymous"

    # Reuse client info from request state if available (set by middleware)
    if hasattr(credentials, "request") and hasattr(
        credentials.request.state, "client_id"
    ):
        return credentials.request.state.client_id

    # Fallback to token validation
    client = key_manager.authenticate_client(credentials.credentials)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    return client.client_id
