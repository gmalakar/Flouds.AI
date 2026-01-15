# =============================================================================
# File: tenant_config_service.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Tenant/global configuration service.

Provides accessors for trusted hosts and CORS origins. Currently reads from
`ConfigLoader.get_app_settings()` so values reflect environment overrides.
Cache can be reset in tests.
"""

from typing import List, Optional

from app.config.config_loader import ConfigLoader
from app.logger import get_logger

logger = get_logger("tenant_config_service")


class TenantConfigService:
    """Simple service to provide tenant and global config values.

    For now this is global-only; it can be extended to read tenant-specific
    config from a DB or management API.
    """

    _cache: Optional[dict[str, List[str]]] = None

    @classmethod
    def _load_settings(cls) -> dict:
        try:
            settings = ConfigLoader.get_app_settings()
            return {
                "trusted_hosts": getattr(settings.security, "trusted_hosts", []) or [],
                "cors_origins": getattr(settings.app, "cors_origins", []) or [],
            }
        except Exception as e:
            logger.error("Failed to load app settings: %s", e)
            return {"trusted_hosts": [], "cors_origins": []}

    @classmethod
    def get_trusted_hosts(cls, tenant_code: str = "") -> List[str]:
        if cls._cache is None:
            cls._cache = cls._load_settings()
        # Future: merge tenant overrides here
        return list(cls._cache.get("trusted_hosts", []))

    @classmethod
    def get_cors_origins(cls, tenant_code: str = "") -> List[str]:
        if cls._cache is None:
            cls._cache = cls._load_settings()
        return list(cls._cache.get("cors_origins", []))

    @classmethod
    def reset_cache(cls):
        cls._cache = None


# singleton
tenant_config_service = TenantConfigService()
