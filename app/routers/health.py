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
from app.utils.memory_monitor import MemoryMonitor

logger = get_logger("health")
router = APIRouter(prefix="/health")


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    uptime_seconds: float
    memory_usage_mb: float
    memory_percent: float


class DetailedHealthResponse(BaseModel):
    status: str
    timestamp: float
    uptime_seconds: float
    memory: Dict[str, Any]
    models: Dict[str, Any]
    disk_space_mb: float


# Track startup time
_startup_time = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    try:
        memory_info = MemoryMonitor.get_memory_info()

        return HealthResponse(
            status="healthy",
            timestamp=time.time(),
            uptime_seconds=time.time() - _startup_time,
            memory_usage_mb=memory_info["rss_mb"],
            memory_percent=memory_info["percent"],
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """Detailed health check with system information."""
    try:
        memory_info = MemoryMonitor.get_memory_info()

        # Check ONNX path availability
        from app.app_init import APP_SETTINGS

        onnx_path = APP_SETTINGS.onnx.onnx_path
        disk_free = 0

        try:
            import shutil

            disk_free = shutil.disk_usage(onnx_path).free / 1024 / 1024
        except:
            disk_free = 0

        # Get model cache info
        model_cache_size = (
            BaseNLPService._model_cache.size()
            if hasattr(BaseNLPService, "_model_cache")
            else 0
        )
        encoder_sessions = (
            BaseNLPService._encoder_sessions.size()
            if hasattr(BaseNLPService._encoder_sessions, "size")
            else 0
        )

        return DetailedHealthResponse(
            status="healthy",
            timestamp=time.time(),
            uptime_seconds=time.time() - _startup_time,
            memory=memory_info,
            models={
                "cached_models": model_cache_size,
                "encoder_sessions": encoder_sessions,
                "onnx_path_exists": os.path.exists(onnx_path),
            },
            disk_space_mb=disk_free,
        )
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    try:
        # Check if ONNX path is accessible
        from app.app_init import APP_SETTINGS

        onnx_path = APP_SETTINGS.onnx.onnx_path

        if not os.path.exists(onnx_path):
            raise HTTPException(status_code=503, detail="ONNX path not accessible")

        return {"status": "ready", "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive", "timestamp": time.time()}
