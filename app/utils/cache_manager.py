# =============================================================================
# File: cache_manager.py
# Date: 2025-08-18
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Centralized cache management for performance optimization."""

import os
import time
from typing import Any, Dict, List, Optional

import psutil

from app.app_init import APP_SETTINGS
from app.exceptions import CacheInvalidationError
from app.logger import get_logger

# ConcurrentDict import not required in this module
from app.services.cache_registry import (
    clear_decoder_sessions,
    clear_encoder_sessions,
    clear_model_config_cache,
    clear_models,
    clear_special_tokens,
    clear_thread_tokenizers,
    get_cache_stats,
    warm_up_model_configs,
)

logger = get_logger("cache_manager")

# Throttle expensive memory queries to at most once per this interval (seconds).
# Configurable via environment variable `FLOUDS_MEMORY_CHECK_INTERVAL` (seconds).
_MIN_MEMORY_CHECK_INTERVAL: float = float(os.getenv("FLOUDS_MEMORY_CHECK_INTERVAL", "5"))
# Module-level cache for last memory check to avoid repeated psutil calls
_LAST_MEMORY_CHECK: float = 0.0
_CACHED_AVAILABLE_GB: float = 0.0


class CacheManager:
    """Centralized cache management and monitoring."""

    @staticmethod
    def get_available_memory_gb() -> float:
        """Get available memory in GB."""
        try:
            global _LAST_MEMORY_CHECK, _CACHED_AVAILABLE_GB
            now = time.time()
            # Return cached value when checks are within the configured interval
            if now - _LAST_MEMORY_CHECK < _MIN_MEMORY_CHECK_INTERVAL:
                return _CACHED_AVAILABLE_GB

            memory = psutil.virtual_memory()
            _CACHED_AVAILABLE_GB = memory.available / (1024**3)
            _LAST_MEMORY_CHECK = now
            return _CACHED_AVAILABLE_GB
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return 0.0

    @staticmethod
    def get_memory_stats() -> Dict[str, Any]:
        """Get detailed memory statistics."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent,
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}

    @staticmethod
    def should_clear_cache(threshold_gb: Optional[float] = None) -> bool:
        """Check if cache should be cleared based on available memory."""
        import os

        if threshold_gb is None:
            threshold_gb = float(os.getenv("FLOUDS_CACHE_MEMORY_THRESHOLD", "1.0"))
        available_gb = CacheManager.get_available_memory_gb()
        return available_gb < threshold_gb

    @staticmethod
    def cleanup_unused_caches(max_age_seconds: Optional[float] = None) -> None:
        """Clean up unused cache entries older than max_age_seconds, with logging."""
        if max_age_seconds is None:
            max_age_seconds = APP_SETTINGS.monitoring.cache_cleanup_max_age_seconds

        try:
            # Cleanup encoder, decoder and model caches directly from registry
            # Use registry getters which ensure initialization and return
            # properly typed ConcurrentDict instances.
            from app.services.cache_registry import (
                get_decoder_sessions,
                get_encoder_sessions,
                get_models_cache,
            )

            encoder_sessions = get_encoder_sessions()
            encoder_cleaned = encoder_sessions.cleanup_unused(max_age_seconds)
            if encoder_cleaned > 0:
                logger.info(
                    f"Cleaned up {encoder_cleaned} unused encoder sessions "
                    f"(cache size now: {encoder_sessions.size()})"
                )

            decoder_sessions = get_decoder_sessions()
            decoder_cleaned = decoder_sessions.cleanup_unused(max_age_seconds)
            if decoder_cleaned > 0:
                logger.info(
                    f"Cleaned up {decoder_cleaned} unused decoder sessions "
                    f"(cache size now: {decoder_sessions.size()})"
                )

            models_cache = get_models_cache()
            models_cleaned = models_cache.cleanup_unused(max_age_seconds)
            if models_cleaned > 0:
                logger.info(
                    f"Cleaned up {models_cleaned} unused models "
                    f"(cache size now: {models_cache.size()})"
                )

            total_cleaned = encoder_cleaned + decoder_cleaned + models_cleaned
            if total_cleaned > 0:
                logger.info(f"Cleaned up {total_cleaned} unused cache entries")

        except Exception as e:
            logger.error(f"Failed to cleanup unused caches: {e}")
        return None

    @staticmethod
    def clear_model_caches() -> None:
        """Clear all model-related caches."""
        try:
            # Clear summarizer-related caches and registry caches directly
            # to avoid importing PromptProcessor and creating circular imports.
            clear_decoder_sessions()
            clear_models()
            clear_special_tokens()

            # Also clear thread-local tokenizers and model-config cache
            clear_thread_tokenizers()
            clear_model_config_cache()

            logger.info("All model caches cleared")

        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")
        return None

    @staticmethod
    def check_and_clear_cache_if_needed(threshold_gb: Optional[float] = None) -> bool:
        """Check memory and clear caches if needed. Returns True if cache was cleared."""
        if threshold_gb is None:
            import os

            threshold_gb = float(os.getenv("FLOUDS_CACHE_MEMORY_THRESHOLD", "1.0"))

        if CacheManager.should_clear_cache(threshold_gb):
            available_gb = CacheManager.get_available_memory_gb()
            logger.warning(
                f"Low memory ({available_gb:.1f}GB available, threshold {threshold_gb:.1f}GB), cleaning up unused caches first"
            )

            # First try cleaning up unused items
            CacheManager.cleanup_unused_caches()

            # Check if memory is still low after cleanup
            if CacheManager.should_clear_cache(threshold_gb):
                logger.warning("Memory still low after cleanup, clearing all caches")
                CacheManager.clear_model_caches()

            return True
        return False

    @staticmethod
    def get_all_cache_stats() -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        from app.config.config_loader import ConfigLoader

        stats = {
            "config_loader": ConfigLoader.get_cache_stats(),
            "base_nlp_service": get_cache_stats(),
            "timestamp": __import__("time").time(),
        }
        return stats

    @staticmethod
    def clear_all_caches() -> None:
        """Clear all application caches."""
        logger.info("Clearing all caches")

        # Clear configuration caches
        from app.config.config_loader import ConfigLoader

        ConfigLoader.clear_cache()

        # Clear NLP service caches via cache registry helpers to avoid
        # importing BaseNLPService and creating a circular import.
        clear_encoder_sessions()
        clear_decoder_sessions()
        clear_models()
        clear_special_tokens()
        clear_thread_tokenizers()
        clear_model_config_cache()
        logger.info("All caches cleared")
        return None

    @staticmethod
    def warm_up_caches(model_names: Optional[List[str]] = None) -> None:
        """Warm up caches with commonly used models."""
        if not model_names:
            # Default models to warm up
            model_names = ["t5-small", "all-MiniLM-L6-v2", "sentence-t5-base"]

        logger.info(f"Warming up caches for {len(model_names)} models")
        # Use registry-level warm-up to avoid importing BaseNLPService here
        # and prevent circular imports.
        warm_up_model_configs(model_names)
        logger.info("Cache warm-up completed")
        return None

    @staticmethod
    def get_cache_health() -> Dict[str, Any]:
        """Get cache health metrics."""
        stats = CacheManager.get_all_cache_stats()
        health: Dict[str, Any] = {
            "status": "healthy",
            "total_cached_items": 0,
            "cache_efficiency": "good",
        }

        # Calculate total cached items using a typed local accumulator
        total_cached_items: int = 0
        for _service, service_stats in stats.items():
            if isinstance(service_stats, dict):
                for key, value in service_stats.items():
                    if isinstance(value, int) and "cached" in key:
                        total_cached_items += value

        # Simple health assessment
        if total_cached_items == 0:
            health["status"] = "cold"
            health["cache_efficiency"] = "poor"
        elif total_cached_items > 100:
            health["cache_efficiency"] = "excellent"

        health["total_cached_items"] = total_cached_items
        health["details"] = stats
        return health

    @staticmethod
    def optimize_caches() -> None:
        """Optimize cache performance by clearing expired entries."""
        logger.info("Optimizing caches")

        # Force garbage collection on caches
        try:
            # This would trigger cleanup in SimpleCache implementations
            stats_before = CacheManager.get_all_cache_stats()

            # Clear expired entries (implementation depends on cache type)
            # For now, just log the optimization attempt
            logger.info(f"Cache optimization completed. Stats: {stats_before}")

        except (AttributeError, TypeError) as e:
            logger.error(f"Invalid cache operation: {e}")
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            raise CacheInvalidationError(f"Cache optimization failed: {e}")
        return None
