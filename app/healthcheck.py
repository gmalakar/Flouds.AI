# =============================================================================
# File: healthcheck.py
# Date: 2025-12-23
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
import sys
from typing import List, Optional
from urllib.parse import urlparse

import requests  # type: ignore[import-untyped]

from app.app_init import APP_SETTINGS


def build_healthcheck_url() -> str:
    # Explicit URL overrides everything if provided
    explicit = os.getenv("HEALTHCHECK_URL")
    if explicit:
        return explicit

    # Compose from host/port/path envs to avoid hardcoding
    host = os.getenv("HEALTHCHECK_HOST", os.getenv("FLOUDS_HOST", "localhost"))
    port = os.getenv("HEALTHCHECK_PORT", os.getenv("FLOUDS_PORT", "19690"))
    path = os.getenv("HEALTHCHECK_PATH", "/api/v1/health")

    # Normalize path
    if not path.startswith("/"):
        path = f"/{path}"

    return f"http://{host}:{port}{path}"


def main() -> int:
    url = build_healthcheck_url()
    timeout_s = float(os.getenv("HEALTHCHECK_TIMEOUT", "8"))
    # Validate URL scheme to allow only http/https
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return 1

    # Build trusted hosts whitelist from settings or env
    def _get_trusted_hosts() -> Optional[List[str]]:
        # Prefer explicit healthcheck env
        env_val = os.getenv("HEALTHCHECK_TRUSTED_HOSTS") or os.getenv("FLOUDS_TRUSTED_HOSTS")
        if env_val:
            return [h.strip() for h in env_val.split(",") if h.strip()]
        try:
            try:
                th = APP_SETTINGS.security
            except Exception:
                th = None
            if th and hasattr(th, "trusted_hosts") and th.trusted_hosts:
                return list(th.trusted_hosts)
        except Exception:
            pass
        return None

    trusted = _get_trusted_hosts()
    host = parsed.hostname or ""
    if trusted and "*" not in trusted:
        # Match host exactly against whitelist
        if host not in trusted:
            return 1

    headers = {"Accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s, allow_redirects=False)
        status = resp.status_code
        if 200 <= int(status) < 400:
            return 0
        return 1
    except requests.RequestException:
        return 1


if __name__ == "__main__":
    sys.exit(main())
