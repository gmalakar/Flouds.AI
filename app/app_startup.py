# =============================================================================
# File: app_startup.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: app_startup.py
# Date: 2026-01-05
# =============================================================================

from contextlib import asynccontextmanager
from os import getenv
from typing import AsyncIterator, List

from fastapi import FastAPI

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.services import config_service
from app.utils.background_cleanup import start_background_cleanup, stop_background_cleanup

# log sanitization not needed in startup

logger = get_logger("app_startup")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan events (extracted from `main.py`).

    Responsibilities:
    - Initialize config service and apply DB settings
    - Seed/override CORS and trusted hosts from env when requested
    - Start the background cleanup monitor (if enabled in settings)
    - Stop the background cleanup monitor on shutdown
    """
    # Startup
    logger.info("Initializing configuration service and applying DB settings")
    try:
        # Ensure config DB/table exists and load runtime settings
        config_service.init_db()
        config_service.load_and_apply_settings()

        # DB-first seeding/override behavior for CORS and Trusted Hosts.
        from typing import Optional

        def _parse_list_env(val: Optional[str]) -> Optional[List[str]]:
            if val is None:
                return None
            if val.strip() == "*":
                return ["*"]
            return [s.strip() for s in val.split(",") if s.strip()]

        cors_env = _parse_list_env(getenv("FLOUDS_CORS_ORIGINS"))
        trusted_env = _parse_list_env(getenv("FLOUDS_TRUSTED_HOSTS"))
        override = getenv("FLOUDS_CONFIG_OVERRIDE") == "1"

        # If override requested, write env -> DB. Otherwise seed DB only when empty.
        try:
            if override:
                if cors_env is not None:
                    config_service.set_cors_origins(cors_env)
                    logger.info("Applied CORS origins from env (override)")
                if trusted_env is not None:
                    config_service.set_trusted_hosts(trusted_env)
                    logger.info("Applied trusted hosts from env (override)")
                config_service.load_and_apply_settings()
            else:
                # Load current DB values to determine if seeding is needed
                existing_cors = config_service.get_cors_origins()
                existing_trusted = config_service.get_trusted_hosts()
                seeded = False
                if (not existing_cors or len(existing_cors) == 0) and cors_env:
                    config_service.set_cors_origins(cors_env)
                    logger.info("Seeded CORS origins from env into DB")
                    seeded = True
                if (not existing_trusted or len(existing_trusted) == 0) and trusted_env:
                    config_service.set_trusted_hosts(trusted_env)
                    logger.info("Seeded trusted hosts from env into DB")
                    seeded = True
                if seeded:
                    config_service.load_and_apply_settings()
                else:
                    # No seeding/override requested — apply whatever is in DB
                    config_service.load_and_apply_settings()
        except Exception:
            logger.exception("Failed to seed/override config from env; falling back to DB values")
    except Exception as e:
        logger.exception("Failed to initialize config service: %s", e)

    logger.info("Starting background cleanup service")
    try:
        if APP_SETTINGS.monitoring.enable_background_cleanup:
            interval = float(APP_SETTINGS.monitoring.background_cleanup_interval_seconds)
            max_age = float(APP_SETTINGS.monitoring.cache_cleanup_max_age_seconds)
            start_background_cleanup(cleanup_interval=interval, max_age_seconds=max_age)
        else:
            logger.info("Background cleanup disabled by configuration (startup)")
    except Exception:
        logger.exception("Failed to start background cleanup service; skipping startup")

    yield

    # Shutdown
    logger.info("Stopping background cleanup service")
    stop_background_cleanup()
