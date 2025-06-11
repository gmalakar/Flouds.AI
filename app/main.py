# =============================================================================
# File: main.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import asyncio
import logging

from app import app
from app.routers import embedder, summarizer
from app.setup import APP_SETTINGS

logger = logging.getLogger("main")

app.include_router(summarizer.router)
app.include_router(embedder.router)


@app.get("/")
def root() -> dict:
    """Root endpoint for health check."""
    return {"message": "Flouds.Py API is running"}


def run_server():
    logger.info(
        f"Starting server: {APP_SETTINGS.server.type} on {APP_SETTINGS.server.host}:{APP_SETTINGS.server.port}"
    )
    server_type = APP_SETTINGS.server.type.lower()
    if server_type == "hypercorn":
        try:
            from hypercorn.asyncio import serve
            from hypercorn.config import Config
        except ImportError:
            logger.error(
                "hypercorn is not installed. Please install it to use this server type."
            )
            raise

        config = Config()
        config.bind = [f"{APP_SETTINGS.server.host}:{APP_SETTINGS.server.port}"]
        config.workers = APP_SETTINGS.server.workers
        config.reload = APP_SETTINGS.server.reload

        asyncio.run(serve(app, config))
    else:
        # Default to uvicorn if unknown type
        import uvicorn

        uvicorn.run(
            app,
            host=APP_SETTINGS.server.host,
            port=APP_SETTINGS.server.port,
            reload=APP_SETTINGS.server.reload,
            workers=APP_SETTINGS.server.workers,
        )


if __name__ == "__main__":
    run_server()
