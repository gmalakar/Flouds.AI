# =============================================================================
# File: cache_manager.py
# Date: 2025-08-18
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Centralized cache management for performance optimization."""

from typing import Any, Dict, List

import psutil

from app.app_init import APP_SETTINGS
from app.exceptions import CacheInvalidationError
from app.logger import get_logger
from app.services.base_nlp_service import BaseNLPService

logger = get_logger("cache_manager")


class CacheManager:
    """Centralized cache management and monitoring."""

    @staticmethod
    def get_available_memory_gb() -> float:
        """Get available memory in GB."""
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024**3)
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
    def should_clear_cache(threshold_gb: float = 1.0) -> bool:
        """Check if cache should be cleared based on available memory."""
        available_gb = CacheManager.get_available_memory_gb()
        return available_gb < threshold_gb

    @staticmethod
    def cleanup_unused_caches(max_age_seconds: float = None):
        """Clean up unused cache entries older than max_age_seconds."""
        if max_age_seconds is None:
            max_age_seconds = APP_SETTINGS.monitoring.cache_cleanup_max_age_seconds

        try:
            from app.services.base_nlp_service import BaseNLPService
            from app.services.summarizer_service import TextSummarizer

            # Cleanup encoder sessions
            encoder_cleaned = BaseNLPService._encoder_sessions.cleanup_unused(
                max_age_seconds
            )

            # Cleanup summarizer sessions if they exist
            decoder_cleaned = 0
            models_cleaned = 0
            if hasattr(TextSummarizer, "_decoder_sessions"):
                decoder_cleaned = TextSummarizer._decoder_sessions.cleanup_unused(
                    max_age_seconds
                )
            if hasattr(TextSummarizer, "_models"):
                models_cleaned = TextSummarizer._models.cleanup_unused(max_age_seconds)

            total_cleaned = encoder_cleaned + decoder_cleaned + models_cleaned
            if total_cleaned > 0:
                logger.info(f"Cleaned up {total_cleaned} unused cache entries")

        except Exception as e:
            logger.error(f"Failed to cleanup unused caches: {e}")

    @staticmethod
    def clear_model_caches():
        """Clear all model-related caches."""
        try:
            # Import here to avoid circular imports
            from app.services.base_nlp_service import BaseNLPService
            from app.services.summarizer_service import TextSummarizer

            # Clear summarizer caches
            TextSummarizer.clear_model_cache()

            # Clear base service caches
            BaseNLPService.clear_encoder_sessions()
            BaseNLPService.clear_thread_tokenizers()
            BaseNLPService.clear_model_config_cache()

            logger.info("All model caches cleared")

        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")

    @staticmethod
    def check_and_clear_cache_if_needed(threshold_gb: float = 1.0) -> bool:
        """Check memory and clear caches if needed. Returns True if cache was cleared."""
        if CacheManager.should_clear_cache(threshold_gb):
            available_gb = CacheManager.get_available_memory_gb()
            logger.warning(
                f"Low memory ({available_gb:.1f}GB available), cleaning up unused caches first"
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
    def get_all_cache_stats() -> Dict[str, any]:
        """Get comprehensive cache statistics."""
        from app.config.config_loader import ConfigLoader

        stats = {
            "config_loader": ConfigLoader.get_cache_stats(),
            "base_nlp_service": BaseNLPService.get_cache_stats(),
            "timestamp": __import__("time").time(),
        }
        return stats

    @staticmethod
    def clear_all_caches():
        """Clear all application caches."""
        logger.info("Clearing all caches")

        # Clear configuration caches
        from app.config.config_loader import ConfigLoader

        ConfigLoader.clear_cache()

        # Clear NLP service caches
        BaseNLPService.clear_encoder_sessions()
        BaseNLPService.clear_thread_tokenizers()
        BaseNLPService.clear_model_config_cache()

        logger.info("All caches cleared")

    @staticmethod
    def warm_up_caches(model_names: List[str] = None):
        """Warm up caches with commonly used models."""
        if not model_names:
            # Default models to warm up
            model_names = ["t5-small", "all-MiniLM-L6-v2", "sentence-t5-base"]

        logger.info(f"Warming up caches for {len(model_names)} models")
        BaseNLPService.warm_up_cache(model_names)
        logger.info("Cache warm-up completed")

    @staticmethod
    def get_cache_health() -> Dict[str, any]:
        """Get cache health metrics."""
        stats = CacheManager.get_all_cache_stats()

        health = {
            "status": "healthy",
            "total_cached_items": 0,
            "cache_efficiency": "good",
        }

        # Calculate total cached items
        for service, service_stats in stats.items():
            if isinstance(service_stats, dict):
                for key, value in service_stats.items():
                    if isinstance(value, int) and "cached" in key:
                        health["total_cached_items"] += value

        # Simple health assessment
        if health["total_cached_items"] == 0:
            health["status"] = "cold"
            health["cache_efficiency"] = "poor"
        elif health["total_cached_items"] > 100:
            health["cache_efficiency"] = "excellent"

        health["details"] = stats
        return health

    @staticmethod
    def optimize_caches():
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
