# =============================================================================
# File: cache_manager.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Centralized cache management for performance optimization."""

from typing import Dict, List

from app.config.config_loader import ConfigLoader
from app.exceptions import CacheInvalidationError
from app.logger import get_logger
from app.services.base_nlp_service import BaseNLPService

logger = get_logger("cache_manager")


class CacheManager:
    """Centralized cache management and monitoring."""

    @staticmethod
    def get_all_cache_stats() -> Dict[str, any]:
        """Get comprehensive cache statistics."""
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
