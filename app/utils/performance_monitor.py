# =============================================================================
# File: performance_monitor.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Optional

import psutil

from app.logger import get_logger

logger = get_logger("performance_monitor")


class PerformanceMonitor:
    """Performance monitoring utilities for tracking system resources and request metrics."""

    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """Get current system performance metrics."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            return {
                "memory": {
                    "total_mb": memory.total / 1024 / 1024,
                    "available_mb": memory.available / 1024 / 1024,
                    "used_mb": memory.used / 1024 / 1024,
                    "percent": memory.percent,
                },
                "cpu": {"percent": cpu_percent, "count": psutil.cpu_count()},
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}

    @staticmethod
    @contextmanager
    def measure_time(operation_name: str):
        """Context manager to measure execution time."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            logger.info(
                f"Performance [{operation_name}]: "
                f"Duration={duration:.3f}s, "
                f"Memory_Delta={memory_delta:.2f}MB"
            )

    @staticmethod
    def performance_decorator(operation_name: Optional[str] = None):
        """Decorator to measure function performance."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                with PerformanceMonitor.measure_time(name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def check_resource_thresholds(
        memory_threshold_mb: int = 1024, cpu_threshold_percent: int = 80
    ) -> Dict[str, bool]:
        """Check if system resources exceed thresholds."""
        try:
            metrics = PerformanceMonitor.get_system_metrics()

            memory_exceeded = (
                metrics.get("memory", {}).get("used_mb", 0) > memory_threshold_mb
            )
            cpu_exceeded = (
                metrics.get("cpu", {}).get("percent", 0) > cpu_threshold_percent
            )

            if memory_exceeded:
                logger.warning(
                    f"Memory usage exceeded threshold: {metrics['memory']['used_mb']:.2f}MB > {memory_threshold_mb}MB"
                )

            if cpu_exceeded:
                logger.warning(
                    f"CPU usage exceeded threshold: {metrics['cpu']['percent']:.1f}% > {cpu_threshold_percent}%"
                )

            return {
                "memory_exceeded": memory_exceeded,
                "cpu_exceeded": cpu_exceeded,
                "healthy": not (memory_exceeded or cpu_exceeded),
            }
        except Exception as e:
            logger.error(f"Failed to check resource thresholds: {e}")
            return {"memory_exceeded": False, "cpu_exceeded": False, "healthy": True}
