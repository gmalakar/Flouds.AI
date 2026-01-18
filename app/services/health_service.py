# =============================================================================
# File: health_service.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
import time
from datetime import datetime
from typing import Any, Dict

from app.exceptions import ComponentHealthError
from app.logger import get_logger
from app.services.base_nlp_service import BaseNLPService
from app.utils.memory_monitor import MemoryMonitor

logger = get_logger("health_service")

# Track service start time
SERVICE_START_TIME = time.time()


class HealthService:
    """Service for health check operations."""

    @classmethod
    def get_health_status(cls) -> Dict:
        """Performs comprehensive health check and returns status."""
        components: Dict[str, str] = {}
        details: Dict[str, Any] = {}

        # Check ONNX models availability
        onnx_status = cls._check_onnx()
        components["onnx"] = onnx_status

        # Check authentication if enabled
        auth_status = cls._check_authentication()
        components["authentication"] = auth_status

        # Check memory status
        memory_status = cls._check_memory()
        components["memory"] = memory_status

        # Calculate uptime
        uptime = time.time() - SERVICE_START_TIME

        # Determine overall status
        overall_status = "healthy"
        if any(status == "unhealthy" for status in components.values()):
            overall_status = "unhealthy"
        elif any(status == "degraded" for status in components.values()):
            overall_status = "degraded"

        return {
            "status": overall_status,
            "service": "Flouds AI",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime,
            "components": components,
            "details": details if details else None,
        }

    @classmethod
    def _check_onnx(cls) -> str:
        """Check ONNX models availability."""
        try:
            from app.app_init import APP_SETTINGS

            onnx_path = getattr(APP_SETTINGS.onnx, "onnx_path", None)
            if not onnx_path or not os.path.exists(onnx_path):
                logger.warning(f"ONNX path not found or not configured: {onnx_path}")
                return "unhealthy"

            # Check if we can access model cache
            if hasattr(BaseNLPService, "_model_cache"):
                logger.debug("ONNX health check passed")
                return "healthy"
            else:
                return "degraded"

        except OSError as e:
            logger.warning(f"ONNX path not accessible: {str(e)}")
            return "unhealthy"
        except Exception as e:
            logger.error(f"ONNX health check failed: {str(e)}")
            raise ComponentHealthError(f"ONNX component health check failed: {e}")

    @classmethod
    def _check_authentication(cls) -> str:
        """Check authentication configuration."""
        try:
            from app.app_init import APP_SETTINGS

            if not APP_SETTINGS.security.enabled:
                return "healthy"  # Auth disabled is valid

            from app.modules.key_manager import key_manager

            if not key_manager.get_all_tokens():
                logger.warning("Authentication enabled but no API keys configured")
                return "unhealthy"

            logger.debug("Authentication health check passed")
            return "healthy"

        except (ImportError, AttributeError) as e:
            logger.warning(f"Authentication module not available: {str(e)}")
            return "unhealthy"
        except Exception as e:
            logger.error(f"Authentication health check failed: {str(e)}")
            raise ComponentHealthError(f"Authentication component health check failed: {e}")

    @classmethod
    def _check_memory(cls) -> str:
        """Check memory usage."""
        try:
            memory_info = MemoryMonitor.get_memory_info()
            memory_percent = memory_info.get("percent", 0)

            if memory_percent > 90:
                logger.warning(f"High memory usage: {memory_percent}%")
                return "unhealthy"
            elif memory_percent > 75:
                logger.warning(f"Elevated memory usage: {memory_percent}%")
                return "degraded"

            logger.debug(f"Memory health check passed: {memory_percent}%")
            return "healthy"

        except (ImportError, AttributeError) as e:
            logger.warning(f"Memory monitoring not available: {str(e)}")
            return "unhealthy"
        except Exception as e:
            logger.error(f"Memory health check failed: {str(e)}")
            raise ComponentHealthError(f"Memory component health check failed: {e}")
