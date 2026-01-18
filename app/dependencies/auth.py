# =============================================================================
# File: auth.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: auth.py
# Repaired middleware implementation using offender_manager
# =============================================================================
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.models.base_response import BaseResponse
from app.modules.key_manager import key_manager
from app.modules.offender_manager import offender_manager
from app.utils.performance_tracker import perf_tracker

logger = get_logger("auth")

router = APIRouter()

# Module-level defaults to avoid function-call defaults in signatures (flake8 B008)
TENANT_HEADER = Header("", alias="X-Tenant-Code", description="Tenant code for request")
SECURITY_DEP = Depends(HTTPBearer(auto_error=False))


def common_headers(tenant_code: str = TENANT_HEADER) -> Dict[str, str]:
    """Dependency that declares common headers used across endpoints.

    This is used for documentation purposes (OpenAPI) so routes show the
    `X-Tenant-Code` header input box in the UI. Authorization is exposed via
    the global HTTPBearer security scheme (the Authorize button) rather than
    a per-endpoint header input to avoid duplication in the docs.
    """
    return {"tenant_code": tenant_code}


# Use the standard HTTPBearer for OpenAPI/dependency injection.
security = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """API Key authentication middleware."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.enabled = APP_SETTINGS.security.enabled

        # Cache valid keys count at startup to avoid repeated calls
        self._keys_configured = bool(key_manager.get_all_tokens()) if self.enabled else True

        self.public_endpoints = frozenset(
            [
                # Avoid overly-broad prefixes that mark entire API as public.
                # Only include specific public endpoints so `/api/v1/*` remains protected.
                "/api/v1/metrics",
                "/favicon.ico",
                # OpenAPI and docs should be public so tooling/tests can fetch specs
                "/openapi.json",
                "/openapi.yaml",
                "/docs",
                "/redoc",
                "/docs/oauth2-redirect",
                # explicit /api/v1 variants
                "/api/v1/openapi.json",
                "/api/v1/openapi.yaml",
                "/api/v1/docs",
                "/api/v1/redoc",
                "/api/v1/docs/oauth2-redirect",
                # health endpoint (explicit)
                "/api/v1/health",
                "/api/v1/health/",
            ]
        )

        # Log security status on startup
        if self.enabled:
            valid_keys = key_manager.get_all_tokens()
            if valid_keys:
                logger.info(f"API authentication enabled with {len(valid_keys)} client(s)")
            else:
                logger.warning("API authentication enabled but no clients configured")
        else:
            logger.info("API authentication disabled")

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request with API key authentication.

        Match behavior with FloudsVector: require `X-Tenant-Code` for all
        non-public endpoints and return a 401 JSON response when missing.
        If authentication is disabled, populate `request.state.tenant_code` and
        allow the request to continue.
        """

        # Determine if authentication checks should be skipped, but still
        # enforce tenant header presence for non-public endpoints.
        skip_auth = not self.enabled

        # Skip auth for public endpoints (optimized lookup)
        path = request.url.path
        # allow exact or prefix matches for entries in the public_endpoints set
        if any(path == pe or path.startswith(pe) for pe in self.public_endpoints):
            return await call_next(request)

        # Require tenant header for all non-public endpoints
        tenant_header = request.headers.get("X-Tenant-Code")

        # Helper to extract client IP
        def _get_client_ip(req: Request) -> str:
            xff = req.headers.get("X-Forwarded-For")
            if xff:
                return xff.split(",", 1)[0].strip()
            client = getattr(req, "client", None)
            try:
                return client.host if client and getattr(client, "host", None) else "unknown"
            except Exception:
                return "unknown"

        client_ip = _get_client_ip(request)

        # Offender check via shared OffenderManager
        try:
            blocked, blocked_until = offender_manager.is_blocked(client_ip)
        except Exception:
            blocked, blocked_until = False, 0.0

        if blocked:
            reason = f"Requests from {client_ip} temporarily blocked until {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(blocked_until))} UTC"
            error_response = BaseResponse(
                success=False,
                message="Blocked due to repeated unauthenticated requests",
                model="auth",
                time_taken=0.0,
                warnings=[reason],
            )
            return JSONResponse(
                status_code=429,
                content=error_response.model_dump(),
                media_type="application/json",
            )

        if not tenant_header:
            # register attempt and possibly block; tenant unknown -> use 'master'
            try:
                blocked_now, reason = offender_manager.register_attempt(client_ip, tenant="master")
            except Exception:
                blocked_now, reason = False, ""

            if blocked_now:
                error_response = BaseResponse(
                    success=False,
                    message="Blocked due to repeated unauthenticated requests",
                    model="auth",
                    time_taken=0.0,
                    warnings=[reason],
                )
                return JSONResponse(
                    status_code=429,
                    content=error_response.model_dump(),
                    media_type="application/json",
                )

            error_response = BaseResponse(
                success=False,
                message="Missing X-Tenant-Code header",
                model="auth",
                time_taken=0.0,
                warnings=[],
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=error_response.model_dump(),
                media_type="application/json",
            )

        # If authentication is disabled, populate request.state and continue.
        if skip_auth:
            request.state.tenant_code = tenant_header
            return await call_next(request)

        # Check if keys are configured (cached check)
        if not self._keys_configured:
            logger.error("Authentication enabled but no API keys configured")
            error_response = BaseResponse(
                success=False,
                message="Authentication misconfigured",
                model="auth",
                time_taken=0.0,
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.model_dump(),
            )

        # Extract and validate token from Authorization header (or query param in dev)
        auth_header = request.headers.get("Authorization")
        token = None

        if auth_header and auth_header.startswith("Bearer ") and len(auth_header) > 7:
            token = auth_header[7:].strip()
        elif not APP_SETTINGS.app.is_production:
            token = request.query_params.get("token")

        if not token:
            error_response = BaseResponse(
                success=False,
                message=(
                    "Missing Authorization header"
                    + (" or token parameter" if not APP_SETTINGS.app.is_production else "")
                ),
                model="auth",
                time_taken=0.0,
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=error_response.model_dump(),
            )

        # Use tenant header when authenticating
        tenant_code = tenant_header or ""
        with perf_tracker.track("auth_client_lookup"):
            client = key_manager.authenticate_client(token, tenant_code)

        if not client:
            logger.warning("Authentication failed for provided API token")
            # Register this failed authentication attempt for offender tracking
            tenant_for_block = tenant_code or "master"
            try:
                blocked_now, reason = offender_manager.register_attempt(
                    client_ip, tenant=tenant_for_block
                )
            except Exception:
                blocked_now, reason = False, ""

            if blocked_now:
                error_response = BaseResponse(
                    success=False,
                    message="Blocked due to repeated unauthenticated requests",
                    model="auth",
                    time_taken=0.0,
                    warnings=[reason],
                )
                return JSONResponse(
                    status_code=429,
                    content=error_response.model_dump(),
                    media_type="application/json",
                )

            error_response = BaseResponse(
                success=False, message="Invalid API token", model="auth", time_taken=0.0
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=error_response.model_dump(),
            )

        # Successful authentication; populate request state
        request.state.client_id = client.client_id
        request.state.client_type = client.client_type
        request.state.tenant_code = tenant_code or getattr(client, "tenant_code", "")

        return await call_next(request)


@router.get("/secure-endpoint")
def secure_endpoint(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = SECURITY_DEP,
) -> dict[str, Any]:
    """Example secured endpoint using dependency injection.

    Prefer the middleware-populated `request.state.client_id` for auth status.
    Fall back to the dependency credentials only when middleware didn't set
    client info (for example in some test flows).
    """
    client_authenticated = False
    if hasattr(request.state, "client_id") and request.state.client_id:
        client_authenticated = True
    elif credentials is not None and getattr(credentials, "credentials", None):
        client_authenticated = True

    return {
        "message": "Secured access granted",
        "client_authenticated": client_authenticated,
    }


# Dependency function for easy reuse
async def get_current_client(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = SECURITY_DEP,
) -> str:
    """Get current authenticated client ID.

    Prefer middleware-populated `request.state.client_id`. If not present,
    fall back to validating the provided credentials (if any).
    """
    if not APP_SETTINGS.security.enabled:
        return "anonymous"

    # Reuse client info from request state if available (set by middleware)
    if hasattr(request.state, "client_id") and request.state.client_id:
        return request.state.client_id

    # Fallback to token validation using dependency-provided credentials
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    tenant_code = request.headers.get("X-Tenant-Code", "")
    client = key_manager.authenticate_client(credentials.credentials, tenant_code)
    if not client:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return client.client_id
