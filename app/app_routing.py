# =============================================================================
# File: app_routing.py
# Date: 2026-01-05
# =============================================================================

"""Centralized router and middleware setup for the FastAPI app.

Move routing and middleware registration out of `main.py` so startup
concerns are easier to test and maintain.
"""
from typing import Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from app.app_init import APP_SETTINGS
from app.dependencies.auth import AuthMiddleware, common_headers
from app.dependencies.auth import router as auth_router
from app.middleware.path_security import PathSecurityMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_validation import RequestValidationMiddleware
from app.middleware.tenant_security import (
    TenantCorsMiddleware,
    TenantTrustedHostMiddleware,
)
from app.routers import (
    admin,
    config,
    embedder,
    extract_embed,
    extractor,
    health,
    model_info,
    rag,
    sendprompt,
    summarizer,
)

# `auth` module object is not imported here; use `auth_router` instead


def setup_routing(app: FastAPI) -> None:
    """Register middleware and API routers on the provided FastAPI app."""

    # Small diagnostic middleware: log requests for /favicon.ico so we can
    # identify the client (useful when a runtime error arises when serving it).
    class FaviconLoggerMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            try:
                if request.url.path == "/favicon.ico":
                    client = request.client.host if request.client else "unknown"
                    ua = request.headers.get("user-agent", "-")
                    # Import logger lazily to avoid import cycles during tests
                    from app.logger import get_logger

                    log = get_logger("favicon_logger")
                    log.info("Favicon request from %s UA=%s", client, ua)
            except Exception:
                pass
            return await call_next(request)

    app.add_middleware(FaviconLoggerMiddleware)

    # Add security middleware
    if APP_SETTINGS.app.is_production:
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=["*"]  # Configure based on deployment
        )

    # Path security (early)
    app.add_middleware(PathSecurityMiddleware)

    # Tenant-aware middleware
    app.add_middleware(TenantCorsMiddleware)
    app.add_middleware(TenantTrustedHostMiddleware)

    # Authentication + request validation
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RequestValidationMiddleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=APP_SETTINGS.app.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    if APP_SETTINGS.rate_limiting.enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=APP_SETTINGS.rate_limiting.requests_per_minute,
            requests_per_hour=APP_SETTINGS.rate_limiting.requests_per_hour,
        )

    # Include routers
    app.include_router(
        summarizer.router,
        prefix="/api/v1/summarizer",
        tags=["Text Summarization"],
        dependencies=[Depends(common_headers)],
    )
    app.include_router(
        embedder.router,
        prefix="/api/v1/embedder",
        tags=["Text Embedding"],
        dependencies=[Depends(common_headers)],
    )
    app.include_router(
        rag.router,
        prefix="/api/v1/rag",
        tags=["RAG (Retrieval-Augmented Generation)"],
        dependencies=[Depends(common_headers)],
    )
    app.include_router(health.router, prefix="/api/v1", tags=["Health & Monitoring"])
    app.include_router(
        admin.router,
        prefix="/api/v1",
        tags=["Administration"],
        dependencies=[Depends(common_headers)],
    )
    app.include_router(
        config.router,
        prefix="/api/v1",
        tags=["Configuration"],
        dependencies=[Depends(common_headers)],
    )
    app.include_router(
        model_info.router,
        prefix="/api/v1",
        tags=["Model Information"],
        dependencies=[Depends(common_headers)],
    )
    app.include_router(
        sendprompt.router,
        prefix="/api/v1",
        tags=["Prompt Processing"],
        dependencies=[Depends(common_headers)],
    )

    app.include_router(
        auth_router,
        prefix="/api/v1/secure-endpoint",
        tags=["Secure Endpoint"],
        dependencies=[Depends(common_headers)],
    )

    app.include_router(
        extractor.router,
        prefix="/api/v1/extractor",
        tags=["Text Extraction"],
        dependencies=[Depends(common_headers)],
    )

    app.include_router(
        extract_embed.router,
        prefix="/api/v1/extract-embed",
        tags=["Extract and Embed"],
        dependencies=[Depends(common_headers)],
    )
