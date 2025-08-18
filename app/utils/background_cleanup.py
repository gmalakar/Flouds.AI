# =============================================================================
# File: background_cleanup.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Background cleanup service for automatic cache management."""

import threading
import time
from typing import Optional

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.utils.cache_manager import CacheManager

logger = get_logger("background_cleanup")


class BackgroundCleanup:
    """Background service for automatic cache cleanup."""

    def __init__(self, cleanup_interval: float = 60.0, max_age_seconds: float = None):
        self.cleanup_interval = cleanup_interval
        self.max_age_seconds = max_age_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the background cleanup service."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Background cleanup started (interval: {self.cleanup_interval}s, max_age: {self.max_age_seconds}s)"
        )

    def stop(self):
        """Stop the background cleanup service."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Background cleanup stopped")

    def _cleanup_loop(self):
        """Main cleanup loop running in background thread."""
        while not self._stop_event.wait(self.cleanup_interval):
            try:
                max_age = self.max_age_seconds
                if max_age is None:
                    max_age = APP_SETTINGS.monitoring.cache_cleanup_max_age_seconds
                CacheManager.cleanup_unused_caches(max_age)
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")


# Global instance
_background_cleanup: Optional[BackgroundCleanup] = None


def start_background_cleanup(
    cleanup_interval: float = 60.0, max_age_seconds: float = None
):
    """Start the global background cleanup service."""
    global _background_cleanup
    if _background_cleanup is None:
        _background_cleanup = BackgroundCleanup(cleanup_interval, max_age_seconds)
    _background_cleanup.start()


def stop_background_cleanup():
    """Stop the global background cleanup service."""
    global _background_cleanup
    if _background_cleanup:
        _background_cleanup.stop()
