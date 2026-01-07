# =============================================================================
# File: health.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.logger import get_logger
from app.services.base_nlp_service import BaseNLPService
from app.services.cache_registry import get_encoder_sessions
from app.services.health_service import HealthService
from app.utils.cache_manager import CacheManager
from app.utils.memory_monitor import MemoryMonitor
from app.utils.performance_tracker import perf_tracker

logger = get_logger("health")
router = APIRouter()


@router.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_status = HealthService.get_health_status()
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system information."""
    try:
        health_status = HealthService.get_health_status()

        # Add detailed system info
        memory_info = MemoryMonitor.get_memory_info()
        from app.app_init import APP_SETTINGS

        onnx_path = getattr(APP_SETTINGS.onnx, "onnx_path", None)

        disk_free = 0
        try:
            import shutil

            if onnx_path:
                disk_free = shutil.disk_usage(onnx_path).free / 1024 / 1024
        except Exception:
            disk_free = 0

        model_cache_size = (
            BaseNLPService._model_cache.size()
            if hasattr(BaseNLPService, "_model_cache")
            else 0
        )
        try:
            encoder_sessions = get_encoder_sessions().size()
        except Exception:
            encoder_sessions = 0

        health_status["memory"] = memory_info
        health_status["models"] = {
            "cached_models": model_cache_size,
            "encoder_sessions": encoder_sessions,
            "onnx_path_exists": os.path.exists(onnx_path) if onnx_path else False,
        }
        health_status["disk_space_mb"] = disk_free

        return health_status
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    try:
        from app.app_init import APP_SETTINGS

        # Only check ONNX path in production
        if APP_SETTINGS.app.is_production:
            onnx_path = getattr(APP_SETTINGS.onnx, "onnx_path", None)
            if not onnx_path or not os.path.exists(onnx_path):
                raise HTTPException(status_code=503, detail="ONNX path not accessible")

        # Check if authentication is properly configured
        if APP_SETTINGS.security.enabled:
            from app.modules.key_manager import key_manager

            if not key_manager.get_all_tokens():
                raise HTTPException(status_code=503, detail="No API keys configured")

        return {"status": "ready", "timestamp": time.time()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive", "timestamp": time.time()}


@router.get("/health/performance")
async def performance_metrics():
    """Performance metrics for rate limiting and authentication."""
    try:
        metrics = {}

        # Get performance stats for key operations
        for operation in [
            "rate_limit_check",
            "rate_limit_cleanup",
            "auth_client_lookup",
        ]:
            stats = perf_tracker.get_stats(operation)
            if stats:
                metrics[operation] = stats

        return {"status": "ok", "timestamp": time.time(), "metrics": metrics}
    except Exception as e:
        logger.error(f"Performance metrics failed: {e}")
        raise HTTPException(status_code=503, detail="Performance metrics unavailable")


@router.get("/health/cache")
async def cache_health():
    """Cache health and statistics."""
    try:
        return CacheManager.get_cache_health()
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        raise HTTPException(status_code=503, detail="Cache health unavailable")


@router.post("/health/cache/clear")
async def clear_caches():
    """Clear all application caches."""
    try:
        CacheManager.clear_all_caches()
        return {
            "status": "success",
            "message": "All caches cleared",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear caches")


@router.post("/health/cache/warmup")
async def warmup_caches():
    """Warm up caches with common models."""
    try:
        CacheManager.warm_up_caches()
        return {
            "status": "success",
            "message": "Cache warm-up completed",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Cache warm-up failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to warm up caches")
