import asyncio
import logging

from app import app
from app.routers import embedder, summarizer
from app.setup import APP_SETTINGS, SERVER_MODULE

logger = logging.getLogger("main")

app.include_router(summarizer.router)
app.include_router(embedder.router)


@app.get("/")
def root() -> dict:
    """Root endpoint for health check."""
    return {"message": "Hello World"}


if __name__ == "__main__":
    logger.info(
        f"Starting server: {APP_SETTINGS.server.type} on {APP_SETTINGS.server.host}:{APP_SETTINGS.server.port}"
    )
    if APP_SETTINGS.server.type.lower() == "uvicorn":
        SERVER_MODULE.run(
            app, host=APP_SETTINGS.server.host, port=APP_SETTINGS.server.port
        )
    elif APP_SETTINGS.server.type.lower() == "hypercorn":
        import asyncio

        asyncio.run(
            SERVER_MODULE.asyncio.serve(
                app,
                config={
                    "bind": f"{APP_SETTINGS.server.host}:{APP_SETTINGS.server.port}"
                },
            )
        )
    else:
        raise ValueError(f"Unsupported server type: {APP_SETTINGS.server.type}")
