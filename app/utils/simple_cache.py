# =============================================================================
# File: simple_cache.py
# Date: 2025-08-01
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from collections import OrderedDict
from threading import Lock
from typing import Any, Optional


class SimpleCache:
    """Simple LRU cache without timers to prevent hanging."""

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                self.cache[key] = value

    def size(self) -> int:
        return len(self.cache)

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
