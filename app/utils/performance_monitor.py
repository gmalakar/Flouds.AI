# =============================================================================
# File: performance_monitor.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import time
from contextlib import contextmanager
from typing import Any, Dict

import psutil

from app.exceptions import InsufficientMemoryError, ResourceException
from app.logger import get_logger
from app.utils.log_sanitizer import sanitize_for_log

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
        except (PermissionError, OSError) as e:
            logger.error("System access denied for metrics: %s", str(e))
            return {}
        except (AttributeError, ImportError) as e:
            logger.error("System monitoring unavailable: %s", str(e))
            return {}
        except Exception as e:
            logger.error("Failed to get system metrics: %s", str(e))
            raise ResourceException(f"System metrics collection failed: {e}")

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
                "Performance [%s]: Duration=%.3fs, Memory_Delta=%.2fMB",
                sanitize_for_log(operation_name),
                duration,
                memory_delta,
            )

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
        except (PermissionError, OSError) as e:
            logger.error("System access denied for threshold check: %s", str(e))
            return {"memory_exceeded": False, "cpu_exceeded": False, "healthy": True}
        except Exception as e:
            logger.error("Failed to check resource thresholds: %s", str(e))
            raise InsufficientMemoryError(f"Resource threshold check failed: {e}")
