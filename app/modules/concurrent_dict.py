# =============================================================================
# File: concurrent_dict.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import time
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple

logger = None  # Will be set by caller to avoid circular imports


class ConcurrentDict:
    """
    Thread-safe dictionary for concurrent access.
    Provides atomic get, set, remove, and get_or_add operations with optional LRU eviction and callbacks.
    """

    def __init__(
        self,
        created_for: Any = None,
        max_size: Optional[int] = None,
        on_evict: Optional[Callable[[Any, Any], None]] = None,
    ):
        self._lock = RLock()
        self._dict = {}
        self._access_times: Dict[Any, float] = {}
        self._created_for = created_for
        self._max_size = max_size if (max_size is None or max_size > 0) else None
        self._on_evict = on_evict  # Optional callback: on_evict(key, value)

    @property
    def created_for(self) -> Any:
        """
        Property to get the 'created_for' attribute.
        """
        return self._created_for

    @created_for.setter
    def created_for(self, value: Any) -> None:
        """
        Property to set the 'created_for' attribute.
        """
        self._created_for = value

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        """
        Thread-safe get operation with access time tracking.
        """
        with self._lock:
            if key in self._dict:
                self._access_times[key] = time.time()
                return self._dict[key]
            return default

    def set(self, key: Any, value: Any) -> None:
        """
        Thread-safe set operation with access time tracking.
        """
        with self._lock:
            self._dict[key] = value
            self._access_times[key] = time.time()
            self._evict_if_needed()

    def remove(self, key: Any) -> None:
        """
        Thread-safe remove operation.
        """
        with self._lock:
            if key in self._dict:
                del self._dict[key]
                self._access_times.pop(key, None)

    def get_or_add(self, key: Any, factory: Callable[[], Any]) -> Any:
        """
        Atomically gets the value for the key, or adds it using the factory if not present.
        """
        with self._lock:
            current_time = time.time()
            if key in self._dict:
                self._access_times[key] = current_time
                return self._dict[key]
            value = factory()
            self._dict[key] = value
            self._access_times[key] = current_time
            self._evict_if_needed()
            return value

    def is_empty(self) -> bool:
        """
        Thread-safe check if the dictionary is empty.
        """
        with self._lock:
            return len(self._dict) == 0

    def size(self) -> int:
        """
        Thread-safe get size of the dictionary.
        """
        with self._lock:
            return len(self._dict)

    def clear(self) -> None:
        """
        Thread-safe clear all items from the dictionary.
        """
        with self._lock:
            self._dict.clear()
            self._access_times.clear()

    def set_max_size(self, max_size: Optional[int]) -> None:
        """Dynamically adjust max size and evict if needed."""
        with self._lock:
            self._max_size = max_size if (max_size is None or max_size > 0) else None
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used items until under max_size, invoking on_evict callback if provided."""
        if not self._max_size:
            return
        # Evict based on oldest access time
        while len(self._dict) > self._max_size and self._access_times:
            lru_key = min(self._access_times, key=self._access_times.get)
            evicted_value = self._dict.pop(lru_key, None)
            self._access_times.pop(lru_key, None)
            # Invoke optional callback for cleanup (e.g., release ONNX sessions)
            if self._on_evict and evicted_value is not None:
                try:
                    self._on_evict(lru_key, evicted_value)
                except Exception as e:
                    if logger:
                        logger.warning(
                            f"Error in on_evict callback for key {lru_key}: {e}"
                        )

    def cleanup_unused(self, max_age_seconds: float = 60.0) -> int:
        """
        Remove items not accessed for max_age_seconds. Returns count of removed items.
        """
        current_time = time.time()
        keys_to_remove = []

        with self._lock:
            for key, access_time in self._access_times.items():
                if current_time - access_time > max_age_seconds:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._dict[key]
                del self._access_times[key]

        return len(keys_to_remove)

    def get_unused_keys(self, max_age_seconds: float = 60.0) -> list:
        """
        Get list of keys not accessed for max_age_seconds.
        """
        current_time = time.time()
        unused_keys = []

        with self._lock:
            for key, access_time in self._access_times.items():
                if current_time - access_time > max_age_seconds:
                    unused_keys.append(key)

        return unused_keys

    @staticmethod
    def add_missing_from_other(
        target: "ConcurrentDict", source: "ConcurrentDict"
    ) -> "ConcurrentDict":
        """
        Thread-safe: Adds only missing key-value pairs from source to target ConcurrentDict.
        If target is None, creates a new ConcurrentDict and copies all items from source.
        Existing keys in target are not overwritten.
        Returns the updated target ConcurrentDict.
        """
        if not isinstance(target, ConcurrentDict) and target is not None:
            raise TypeError("target must be a ConcurrentDict or None")
        if not isinstance(source, ConcurrentDict):
            raise TypeError("source must be a ConcurrentDict")
        if target is None:
            target = ConcurrentDict()
        with target._lock, source._lock:
            for key, value in source._dict.items():
                if key not in target._dict:
                    target._dict[key] = value
        return target
