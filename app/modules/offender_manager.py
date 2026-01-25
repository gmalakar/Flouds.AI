# =============================================================================
# File: offender_manager.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: offender_manager.py
# In-memory offender tracker for repeated unauthenticated requests.
# =============================================================================
import os
import threading
import time
from typing import Tuple

from app.logger import get_logger
from app.services.config_service import get_config, set_config

logger = get_logger("offender_manager")


class OffenderManager:
    """Simple in-memory offender tracker keyed by client IP.

    Responsibilities:
    - Track failed/anonymous request attempts per-IP.
    - Determine whether an IP is currently blocked.
    - Load per-tenant (or global) blocking thresholds from config DB,
      falling back to env vars and hard-coded defaults.

    Note: This is intentionally process-local. For multi-process deployments
    use a shared store (Redis) instead.
    """

    def __init__(self) -> None:
        # ip -> {count, first_seen, blocked_until}
        self._offender_store: dict[str, dict] = {}
        self._offender_lock = threading.Lock()
        # tenant_code ("" for global) -> (max_attempts, window_seconds, block_seconds)
        self._tenant_block_config: dict[str, tuple[int, int, int]] = {}

        # Ensure master/global defaults exist in config DB
        try:
            defaults = {
                "block_max_attempts": "5",
                "block_window_seconds": "60",
                "block_seconds": "200",
            }
            for k, v in defaults.items():
                cur = get_config(k, "master")
                if cur is None:
                    set_config(k, v, "master")
                    logger.info(f"Inserted default config {k}={v} for tenant 'master'")
        except Exception:
            logger.exception("Failed to ensure default block config in config DB")

    def _get_block_config_for_tenant(self, tenant: str) -> tuple[int, int, int]:
        t = tenant or ""
        if t in self._tenant_block_config:
            return self._tenant_block_config[t]

        def _val_from_config_or_env(key: str, env_name: str, default: int) -> int:
            try:
                if t != "":
                    v = get_config(key, t)
                    if v is not None:
                        return int(v)
            except Exception:
                pass
            # Try global-scoped config
            try:
                gv = get_config(key, "")
                if gv is not None:
                    return int(gv)
            except Exception:
                pass
            try:
                ev = os.getenv(env_name)
                if ev is not None:
                    return int(ev)
            except Exception:
                pass
            return default

        max_attempts = _val_from_config_or_env(
            "block_max_attempts", "FLOUDS_BLOCK_MAX_ATTEMPS", 5
        )
        window_seconds = _val_from_config_or_env(
            "block_window_seconds", "FLOUDS_BLOCK_WINDOW_SECONDS", 60
        )
        block_seconds = _val_from_config_or_env(
            "block_seconds", "FLOUDS_BLOCK_SECONDS", 200
        )

        cfg = (max_attempts, window_seconds, block_seconds)
        self._tenant_block_config[t] = cfg
        return cfg

    def is_blocked(self, ip: str) -> Tuple[bool, float]:
        now = time.time()
        with self._offender_lock:
            rec = self._offender_store.get(ip)
            if rec and rec.get("blocked_until", 0) > now:
                return True, rec.get("blocked_until", 0)
        return False, 0.0

    def register_attempt(self, ip: str, tenant: str = "") -> Tuple[bool, str]:
        """Register a failed attempt. Return (blocked_now, reason).

        Uses tenant (empty string means global). If threshold exceeded,
        sets blocked_until and returns True.
        """
        now = time.time()
        max_attempts, window_seconds, block_seconds = self._get_block_config_for_tenant(
            tenant
        )
        with self._offender_lock:
            rec = self._offender_store.get(ip)
            if not rec:
                self._offender_store[ip] = {
                    "count": 1,
                    "first_seen": now,
                    "blocked_until": 0,
                }
                return False, ""

            if now - rec["first_seen"] > window_seconds:
                rec["count"] = 1
                rec["first_seen"] = now
                rec["blocked_until"] = 0
                return False, ""

            rec["count"] += 1
            if rec["count"] > max_attempts:
                rec["blocked_until"] = now + block_seconds
                reason = f"Blocked until {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(rec['blocked_until']))} UTC"
                return True, reason
            return False, ""


offender_manager = OffenderManager()
