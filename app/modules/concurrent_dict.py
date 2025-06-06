import threading
from typing import Any, Callable, Optional


class ConcurrentDict:
    """
    Thread-safe dictionary for concurrent access.
    Provides atomic get, set, remove, and get_or_add operations.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._dict = {}

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
