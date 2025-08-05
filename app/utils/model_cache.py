# =============================================================================
# File: model_cache.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import time
from collections import OrderedDict
from threading import Lock, Timer
from typing import Any, Optional

# Removed logger to prevent hanging


class LRUModelCache:
    """Thread-safe LRU cache for models with sliding expiration."""

    def __init__(self, max_size: int = 5, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.access_times = {}
        self.timers = {}  # Track cleanup timers
        self.lock = Lock()

        # Start periodic cleanup
        self._start_cleanup_timer()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, updating access time and resetting expiration."""
        with self.lock:
            if key not in self.cache:
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()

            # Reset sliding expiration timer
            self._reset_expiration_timer(key)

            return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        """Add item to cache, evicting LRU if needed."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    lru_key = next(iter(self.cache))
                    self._remove(lru_key)
                    pass  # Evicted model from cache

                self.cache[key] = value
                pass  # Added model to cache

            self.access_times[key] = time.time()
            self._reset_expiration_timer(key)

    def _remove(self, key: str) -> None:
        """Remove item from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]

            # Cancel expiration timer
            if key in self.timers:
                self.timers[key].cancel()
                del self.timers[key]

    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            # Cancel all timers
            for timer in self.timers.values():
                timer.cancel()

            self.cache.clear()
            self.access_times.clear()
            self.timers.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def _reset_expiration_timer(self, key: str) -> None:
        """Reset sliding expiration timer for a key."""
        # Cancel existing timer
        if key in self.timers:
            self.timers[key].cancel()

        # Create new timer
        timer = Timer(self.ttl_seconds, self._expire_key, args=[key])
        self.timers[key] = timer
        timer.start()

    def _expire_key(self, key: str) -> None:
        """Expire a key due to inactivity."""
        with self.lock:
            if key in self.cache:
                pass  # Model expired due to inactivity
                self._remove(key)

    def _start_cleanup_timer(self) -> None:
        """Start periodic cleanup of expired items."""

        def cleanup():
            with self.lock:
                current_time = time.time()
                expired_keys = [
                    key
                    for key, access_time in self.access_times.items()
                    if current_time - access_time > self.ttl_seconds
                ]

                for key in expired_keys:
                    pass  # Cleaning up expired model
                    self._remove(key)

            # Schedule next cleanup
            Timer(self.ttl_seconds // 2, cleanup).start()

        # Start first cleanup
        Timer(self.ttl_seconds // 2, cleanup).start()
