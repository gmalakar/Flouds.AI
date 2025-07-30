# =============================================================================
# File: memory_monitor.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

import gc
from typing import Any, Dict

from app.logger import get_logger

logger = get_logger("memory_monitor")


class MemoryMonitor:
    """Monitor and manage memory usage."""

    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get current memory usage information."""
        if not PSUTIL_AVAILABLE:
            return {"rss_mb": 0.0, "vms_mb": 0.0, "percent": 0.0, "available_mb": 0.0}

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    @staticmethod
    def check_memory_threshold(threshold_mb: int = 1000) -> bool:
        """Check if memory usage exceeds threshold."""
        if not PSUTIL_AVAILABLE:
            return False
        memory_info = MemoryMonitor.get_memory_info()
        return memory_info["rss_mb"] > threshold_mb

    @staticmethod
    def force_garbage_collection() -> Dict[str, int]:
        """Force garbage collection and return stats."""
        before = MemoryMonitor.get_memory_info()

        # Force garbage collection
        collected = gc.collect()

        after = MemoryMonitor.get_memory_info()

        freed_mb = before["rss_mb"] - after["rss_mb"]

        logger.info(f"GC collected {collected} objects, freed {freed_mb:.2f} MB")

        return {
            "objects_collected": collected,
            "memory_freed_mb": freed_mb,
            "memory_before_mb": before["rss_mb"],
            "memory_after_mb": after["rss_mb"],
        }

    @staticmethod
    def log_memory_usage(context: str = "") -> None:
        """Log current memory usage."""
        memory_info = MemoryMonitor.get_memory_info()
        logger.info(
            f"Memory usage {context}: {memory_info['rss_mb']:.2f} MB ({memory_info['percent']:.1f}%)"
        )
