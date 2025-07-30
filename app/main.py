# =============================================================================
# File: main.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import signal
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.middleware.auth import AuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_validation import RequestValidationMiddleware
from app.routers import admin, embedder, health, summarizer

logger = get_logger("main")

app = FastAPI(
    title=APP_SETTINGS.app.name,
    description=APP_SETTINGS.app.description,
    version=APP_SETTINGS.app.version,
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
)

# Add security middleware
if APP_SETTINGS.app.is_production:
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["*"]  # Configure based on your deployment
    )

# Add authentication middleware
app.add_middleware(AuthMiddleware)

# Add request validation middleware
app.add_middleware(RequestValidationMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=APP_SETTINGS.app.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
if APP_SETTINGS.rate_limiting.enabled:
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=APP_SETTINGS.rate_limiting.requests_per_minute,
        requests_per_hour=APP_SETTINGS.rate_limiting.requests_per_hour,
    )
app.include_router(
    summarizer.router, prefix="/api/v1/summarizer", tags=["Text Summarization"]
)
app.include_router(embedder.router, prefix="/api/v1/embedder", tags=["Text Embedding"])
app.include_router(health.router, prefix="/api/v1", tags=["Health & Monitoring"])
app.include_router(admin.router, prefix="/api/v1", tags=["Administration"])


@app.get("/")
def root() -> dict:
    """Root endpoint for health check."""
    return {
        "message": "Flouds AI API is running",
        "version": "v1",
        "docs": "/api/v1/docs",
    }


@app.get("/api/v1")
def api_v1_root() -> dict:
    """API v1 root endpoint."""
    return {"message": "Flouds AI API v1", "version": "v1", "docs": "/api/v1/docs"}


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


def run_server():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(
        f"Starting uvicorn server on {APP_SETTINGS.server.host}:{APP_SETTINGS.server.port}"
    )

    uvicorn.run(
        "app.main:app",
        host=APP_SETTINGS.server.host,
        port=APP_SETTINGS.server.port,
        workers=None,
        reload=not APP_SETTINGS.app.is_production,
        log_level="info" if not APP_SETTINGS.app.debug else "debug",
        access_log=True,
        timeout_keep_alive=APP_SETTINGS.server.keepalive_timeout,
        timeout_graceful_shutdown=APP_SETTINGS.server.graceful_timeout,
    )


if __name__ == "__main__":
    try:
        run_server()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error("Fatal error:", exc_info=e)
        sys.exit(1)

# Run Instruction
# Unit Test : python -m pytest
# Run for terminal: python -m app.main
