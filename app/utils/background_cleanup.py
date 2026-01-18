# =============================================================================
# File: background_cleanup.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Background cleanup service for automatic cache management."""

import atexit
import random
import threading
import traceback
from typing import Optional

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.utils.cache_manager import CacheManager

logger = get_logger("background_cleanup")


class BackgroundCleanup:
    """Background service for automatic cache cleanup."""

    def __init__(self, cleanup_interval: float = 60.0, max_age_seconds: Optional[float] = None):
        self.cleanup_interval = cleanup_interval
        self.max_age_seconds = max_age_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        # Backoff state for repeated errors
        self._consecutive_errors = 0
        self._current_backoff = 0.0

    def start(self):
        """Start the background cleanup service."""
        if self._running:
            return

        # Respect global enable flag from settings
        if not APP_SETTINGS.monitoring.enable_background_cleanup:
            logger.info("Background cleanup disabled by configuration")
            return

        self._running = True
        self._stop_event.clear()
        # reset backoff state
        self._consecutive_errors = 0
        self._current_backoff = 0.0
        self._thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Background cleanup monitor started (interval: {self.cleanup_interval}s, max_age: {self.max_age_seconds}s)"
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
        # Apply an initial randomized jitter so multiple instances don't run simultaneously
        try:
            initial_jitter = float(
                APP_SETTINGS.monitoring.background_cleanup_initial_jitter_seconds
            )
        except Exception:
            initial_jitter = 0.0

        if initial_jitter and not self._stop_event.is_set():
            delay = random.uniform(0, initial_jitter)
            logger.debug(f"Background cleanup initial jitter delay: {delay:.2f}s")
            if self._stop_event.wait(delay):
                return

        # Main loop
        while not self._stop_event.is_set():
            try:
                max_age = self.max_age_seconds
                if max_age is None:
                    max_age = APP_SETTINGS.monitoring.cache_cleanup_max_age_seconds

                CacheManager.cleanup_unused_caches(max_age)

                # Success -> reset backoff
                self._consecutive_errors = 0
                self._current_backoff = 0.0

                # Wait for the configured interval (allow early exit)
                interval = (
                    self.cleanup_interval
                    or APP_SETTINGS.monitoring.background_cleanup_interval_seconds
                )
                if self._stop_event.wait(interval):
                    break

            except Exception as e:
                # Increment error counter and compute exponential backoff
                self._consecutive_errors += 1
                try:
                    max_backoff = float(
                        APP_SETTINGS.monitoring.background_cleanup_max_backoff_seconds
                    )
                except Exception:
                    max_backoff = 300.0

                # Exponential backoff: min(max_backoff, 2^n * base_interval)
                base = (
                    self.cleanup_interval
                    or APP_SETTINGS.monitoring.background_cleanup_interval_seconds
                    or 60
                )
                backoff = min(max_backoff, (2**self._consecutive_errors) * float(base))
                # add some jitter to backoff
                jitter = random.uniform(0, min(5.0, backoff * 0.1))
                self._current_backoff = backoff + jitter

                logger.error(
                    "Background cleanup error (attempt %d): %s; backing off %.1fs",
                    self._consecutive_errors,
                    str(e),
                    self._current_backoff,
                )
                logger.debug(traceback.format_exc())

                # Wait for backoff period (allow stop)
                if self._stop_event.wait(self._current_backoff):
                    break


# Global instance
_background_cleanup: Optional[BackgroundCleanup] = None


def start_background_cleanup(
    cleanup_interval: float = 60.0, max_age_seconds: Optional[float] = None
) -> None:
    """Start the global background cleanup service."""
    global _background_cleanup
    if _background_cleanup is None:
        _background_cleanup = BackgroundCleanup(cleanup_interval, max_age_seconds)
    _background_cleanup.start()


def stop_background_cleanup() -> None:
    """Stop the global background cleanup service."""
    global _background_cleanup
    if _background_cleanup:
        _background_cleanup.stop()


# Ensure background cleanup stops on process exit
atexit.register(lambda: stop_background_cleanup())
