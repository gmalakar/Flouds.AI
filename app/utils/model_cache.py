# =============================================================================
# File: model_cache.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

from app.logger import get_logger

logger = get_logger("model_cache")


class LRUModelCache:
    """Thread-safe LRU cache for models with size limits."""

    def __init__(self, max_size: int = 5, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.access_times = {}
        self.lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, updating access time."""
        with self.lock:
            if key not in self.cache:
                return None

            # Check TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                self._remove(key)
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
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
                    logger.info(f"Evicted model from cache: {lru_key}")

                self.cache[key] = value
                logger.info(f"Added model to cache: {key}")

            self.access_times[key] = time.time()

    def _remove(self, key: str) -> None:
        """Remove item from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]

    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
