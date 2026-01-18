# =============================================================================
# File: main.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
import signal
import sys
import warnings
from types import FrameType
from typing import Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer

from app.app_init import APP_SETTINGS
from app.app_routing import setup_routing
from app.app_startup import lifespan
from app.exceptions import FloudsBaseException
from app.logger import get_logger
from app.utils.enhance_openapi import setup_enhanced_openapi
from app.utils.error_handler import ErrorHandler
from app.utils.log_sanitizer import sanitize_for_log

# Suppress PyTorch ONNX warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")

logger = get_logger("main")


# lifespan extracted to `app.app_startup.lifespan`


app = FastAPI(
    title=APP_SETTINGS.app.name,
    description=APP_SETTINGS.app.description,
    version=APP_SETTINGS.app.version,
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    lifespan=lifespan,
    # Keep a non-blocking HTTPBearer at app-level so the OpenAPI "Authorize"
    # control is available, but don't add `common_headers` here because it
    # injects header parameters into every endpoint (including public ones
    # like `/health`). Attach `common_headers` only to secured routers below.
    dependencies=[Depends(HTTPBearer(auto_error=False))],
)


# Use centralized enhanced OpenAPI generator so docs/metadata match
# the FloudsVector project. This replaces the ad-hoc `_custom_openapi`.
setup_enhanced_openapi(app)


# Ensure a named HTTP Bearer security scheme is present in OpenAPI so Swagger
# Authorize control consistently sets the Authorization header when used.


# Keep non-blocking HTTPBearer at app-level so OpenAPI shows Authorize
# app.dependency_overrides = getattr(app, "dependency_overrides", {})
# app_deps = [Depends(HTTPBearer(auto_error=False))]
# app.dependencies = getattr(app, "dependencies", []) + app_deps


# Global exception handlers
@app.exception_handler(FloudsBaseException)
async def flouds_exception_handler(request: Request, exc: FloudsBaseException) -> JSONResponse:
    """Handle custom Flouds exceptions."""
    status_code = ErrorHandler.get_http_status(exc)
    logger.warning(
        "Flouds exception in %s: %s",
        sanitize_for_log(str(request.url)),
        sanitize_for_log(exc.message),
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": exc.message,
            "error_code": exc.error_code,
            "detail": exc.message,
        },
    )


@app.exception_handler(MemoryError)
async def memory_error_handler(request: Request, exc: MemoryError) -> JSONResponse:
    """Handle out-of-memory errors without crashing the service."""
    logger.error(
        "Out of memory error in %s: %s",
        sanitize_for_log(str(request.url)),
        sanitize_for_log(str(exc)),
    )
    # Trigger cache cleanup
    try:
        from app.utils.cache_manager import CacheManager

        CacheManager.clear_all_caches()
        logger.info("Cleared caches after memory error")
    except Exception as cleanup_err:
        logger.warning(f"Cache cleanup failed: {cleanup_err}")

    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "message": "Out of memory: request too large or system resources exhausted",
            "error_code": "OUT_OF_MEMORY",
            "detail": "Please reduce input size or try again later",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    # Let the ErrorHandler process the exception (logging/telemetry)
    ErrorHandler.handle_exception(exc, f"request to {request.url}", include_traceback=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "detail": "An unexpected error occurred",
        },
    )


# Configure middleware and routers (moved to app/app_routing.py)
# Centralized routing + middleware setup
setup_routing(app)


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint for health check."""
    return {
        "message": "Flouds AI API is running",
        "version": "v1",
        "docs": "/api/v1/docs",
    }


@app.get("/api/v1")
def api_v1_root() -> dict[str, str]:
    """API v1 root endpoint."""
    return {"message": "Flouds AI API v1", "version": "v1", "docs": "/api/v1/docs"}


@app.get("/favicon.ico")
def favicon() -> Response:
    """Return empty response for favicon requests."""
    # 204 must not include a response body; return a bare Response to avoid
    # mismatches between Content-Length and actual body length.
    return Response(status_code=204)


def cleanup_handlers():
    """Clean up logger handlers."""
    import logging

    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
    logging.shutdown()


def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    cleanup_handlers()
    sys.exit(0)


def run_server():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(f"Starting uvicorn server on {APP_SETTINGS.server.host}:{APP_SETTINGS.server.port}")

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=APP_SETTINGS.server.host,
        port=APP_SETTINGS.server.port,
        workers=None,
        reload=not APP_SETTINGS.app.is_production,
        log_level="debug" if os.getenv("APP_DEBUG_MODE", "0") == "1" else "info",
        access_log=True,
        timeout_keep_alive=APP_SETTINGS.server.keepalive_timeout,
        timeout_graceful_shutdown=APP_SETTINGS.server.graceful_timeout,
    )

    logger.info("Flouds AI server stopped")


if __name__ == "__main__":
    try:
        print("Starting Flouds AI application...")
        run_server()
    except KeyboardInterrupt:
        print("Application stopped by user")
        logger.info("Application stopped by user")
    except MemoryError as e:
        print(f"Out of memory error: {e}")
        logger.error("Out of memory error:", exc_info=e)
        sys.exit(137)  # Docker OOM exit code
    except Exception as e:
        print(f"Fatal error during startup: {e}")
        logger.error("Fatal error:", exc_info=e)
        import traceback

        traceback.print_exc()
        sys.exit(1)

# Run Instruction
# Set Env: $env:FLOUDS_API_ENV="Development"
# Unit Test : python -m pytest
# Run for terminal: python -m app.main
