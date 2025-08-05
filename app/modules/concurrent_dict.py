# =============================================================================
# File: concurrent_dict.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from threading import RLock
from typing import Any, Callable, Optional


class ConcurrentDict:
    """
    Thread-safe dictionary for concurrent access.
    Provides atomic get, set, remove, and get_or_add operations.
    """

    def __init__(self, created_for: Any = None):
        self._lock = RLock()
        self._dict = {}
        self._created_for = created_for

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
        Thread-safe get operation.
        """
        with self._lock:
            return self._dict.get(key, default)

    def set(self, key: Any, value: Any) -> None:
        """
        Thread-safe set operation.
        """
        with self._lock:
            self._dict[key] = value

    def remove(self, key: Any) -> None:
        """
        Thread-safe remove operation.
        """
        with self._lock:
            if key in self._dict:
                del self._dict[key]

    def get_or_add(self, key: Any, factory: Callable[[], Any]) -> Any:
        """
        Atomically gets the value for the key, or adds it using the factory if not present.
        """
        with self._lock:
            if key in self._dict:
                return self._dict[key]
            value = factory()
            self._dict[key] = value
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
