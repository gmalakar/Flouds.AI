# =============================================================================
# File: healthcheck.py
# Date: 2025-12-23
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
import sys
import urllib.error
import urllib.request


def build_healthcheck_url() -> str:
    # Explicit URL overrides everything if provided
    explicit = os.getenv("HEALTHCHECK_URL")
    if explicit:
        return explicit

    # Compose from host/port/path envs to avoid hardcoding
    host = os.getenv("HEALTHCHECK_HOST", os.getenv("SERVER_HOST", "localhost"))
    port = os.getenv("HEALTHCHECK_PORT", os.getenv("SERVER_PORT", "19690"))
    path = os.getenv("HEALTHCHECK_PATH", "/api/v1/health")

    # Normalize path
    if not path.startswith("/"):
        path = f"/{path}"

    return f"http://{host}:{port}{path}"


def main() -> int:
    url = build_healthcheck_url()
    timeout_s = float(os.getenv("HEALTHCHECK_TIMEOUT", "8"))

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            # Consider any 2xx/3xx as healthy
            if 200 <= status < 400:
                return 0
            else:
                return 1
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception):
        return 1


if __name__ == "__main__":
    sys.exit(main())
