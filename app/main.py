# =============================================================================
# File: main.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import signal
import sys

import uvicorn
from fastapi import FastAPI

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.routers import embedder, summarizer

logger = get_logger("main")

app = FastAPI(
    title="Flouds AI API",
    description="API for Flouds AI, a cloud-based summarization and embedding service.",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.include_router(summarizer.router)
app.include_router(embedder.router)


@app.get("/")
def root() -> dict:
    """Root endpoint for health check."""
    return {"message": "Flouds AI API is running"}


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "Flouds AI"}


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
    )


if __name__ == "__main__":
    try:
        run_server()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error("Fatal error:", exc_info=e)
        sys.exit(1)
