# =============================================================================
# File: performance_tracker.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Dict, Optional

from app.logger import get_logger

logger = get_logger("performance_tracker")


class PerformanceTracker:
    """Track performance metrics for rate limiting and authentication."""

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.counters: Dict[str, int] = defaultdict(int)

    @contextmanager
    def track(self, operation: str):
        """Context manager to track operation timing."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.metrics[operation].append(duration)
            self.counters[operation] += 1

    def get_stats(self, operation: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return None

        times = list(self.metrics[operation])
        return {
            "count": len(times),
            "avg_ms": sum(times) * 1000 / len(times),
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "total_calls": self.counters[operation],
        }

    def log_stats(self):
        """Log performance statistics."""
        for operation in self.metrics:
            stats = self.get_stats(operation)
            if stats:
                logger.info(
                    f"Performance [{operation}]: "
                    f"avg={stats['avg_ms']:.2f}ms, "
                    f"min={stats['min_ms']:.2f}ms, "
                    f"max={stats['max_ms']:.2f}ms, "
                    f"calls={stats['total_calls']}"
                )


# Global performance tracker
perf_tracker = PerformanceTracker()
