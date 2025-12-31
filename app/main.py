# =============================================================================
# File: main.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
import signal
import sys
import warnings
from contextlib import asynccontextmanager

from app.middleware import auth

# Suppress PyTorch ONNX warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")

print("Starting imports...")
import gc

from fastapi import FastAPI, Request

print("FastAPI imported")

# Force garbage collection to free memory
gc.collect()
print("Memory cleanup completed")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.app_init import APP_SETTINGS
from app.exceptions import FloudsBaseException
from app.logger import get_logger
from app.middleware.auth import AuthMiddleware
from app.middleware.path_security import PathSecurityMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_validation import RequestValidationMiddleware
from app.routers import (
    admin,
    embedder,
    extract_embed,
    extractor,
    health,
    model_info,
    rag,
    sendprompt,
    summarizer,
)
from app.utils.background_cleanup import (
    start_background_cleanup,
    stop_background_cleanup,
)
from app.utils.error_handler import ErrorHandler
from app.utils.log_sanitizer import sanitize_for_log

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting background cleanup service")
    start_background_cleanup(cleanup_interval=60.0, max_age_seconds=60.0)

    yield

    # Shutdown
    logger.info("Stopping background cleanup service")
    stop_background_cleanup()


app = FastAPI(
    title=APP_SETTINGS.app.name,
    description=APP_SETTINGS.app.description,
    version=APP_SETTINGS.app.version,
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    lifespan=lifespan,
)


# Global exception handlers
@app.exception_handler(FloudsBaseException)
async def flouds_exception_handler(request: Request, exc: FloudsBaseException):
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
async def memory_error_handler(request: Request, exc: MemoryError):
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
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    error_response = ErrorHandler.handle_exception(
        exc, f"request to {request.url}", include_traceback=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "detail": "An unexpected error occurred",
        },
    )


# Add security middleware
if APP_SETTINGS.app.is_production:
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["*"]  # Configure based on your deployment
    )

# Add path security middleware (first for early detection)
app.add_middleware(PathSecurityMiddleware)

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

# Include routers
app.include_router(
    summarizer.router, prefix="/api/v1/summarizer", tags=["Text Summarization"]
)
app.include_router(embedder.router, prefix="/api/v1/embedder", tags=["Text Embedding"])
app.include_router(
    rag.router, prefix="/api/v1/rag", tags=["RAG (Retrieval-Augmented Generation)"]
)
app.include_router(health.router, prefix="/api/v1", tags=["Health & Monitoring"])
app.include_router(admin.router, prefix="/api/v1", tags=["Administration"])
app.include_router(model_info.router, prefix="/api/v1", tags=["Model Information"])
app.include_router(sendprompt.router, prefix="/api/v1", tags=["Prompt Processing"])

app.include_router(
    auth.router, prefix="/api/v1/secure-endpoint", tags=["Secure Endpoint"]
)

app.include_router(
    extractor.router, prefix="/api/v1/extractor", tags=["Text Extraction"]
)

app.include_router(
    extract_embed.router,
    prefix="/api/v1/extract-embed",
    tags=["Extract and Embed"],
)


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


@app.get("/favicon.ico")
def favicon():
    """Return empty response for favicon requests."""
    from fastapi.responses import Response

    return Response(status_code=204)


def cleanup_handlers():
    """Clean up logger handlers."""
    import logging

    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
    logging.shutdown()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    cleanup_handlers()
    sys.exit(0)


def run_server():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(
        f"Starting uvicorn server on {APP_SETTINGS.server.host}:{APP_SETTINGS.server.port}"
    )

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
