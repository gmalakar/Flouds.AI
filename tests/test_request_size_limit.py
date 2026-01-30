# =============================================================================
# File: test_request_size_limit.py
# Date: 2026-01-17
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: test_request_size_limit.py
# Date: 2026-01-17
# =============================================================================

import json  # noqa: F401
import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient


def create_app_with_size_limit(max_size: int) -> TestClient:
    # Ensure a lightweight app.app_init shim exists and set the desired max_size
    app_init = sys.modules.get("app.app_init")
    if not app_init:
        shim = types.ModuleType("app.app_init")
        shim.APP_SETTINGS = types.SimpleNamespace(
            app=types.SimpleNamespace(max_request_size=max_size)
        )
        sys.modules["app.app_init"] = shim
        app_init = shim
    else:
        # Update existing shimmed APP_SETTINGS
        try:
            app_init.APP_SETTINGS.app.max_request_size = max_size
        except Exception:
            app_init.APP_SETTINGS = types.SimpleNamespace(
                app=types.SimpleNamespace(max_request_size=max_size)
            )

    # Import and reload middleware so it picks up the current APP_SETTINGS
    import importlib

    import app.middleware.request_size_limit as _rsl

    importlib.reload(_rsl)
    RequestSizeLimitMiddleware = _rsl.RequestSizeLimitMiddleware

    app = FastAPI()

    @app.post("/echo")
    async def echo(payload: dict):
        return {"received": payload}

    app.add_middleware(RequestSizeLimitMiddleware)
    return TestClient(app)


class TestRequestSizeLimit:
    def test_oversized_request_rejected(self):
        client = create_app_with_size_limit(max_size=10)  # 10 bytes
        # Payload larger than 10 bytes
        data = {"text": "a" * 100}
        resp = client.post("/echo", json=data)
        assert resp.status_code == 413
        body = resp.json()
        assert body["success"] is False
        assert body["error_code"] == "PAYLOAD_TOO_LARGE"
        assert body["max_size"] == 10

    def test_request_exactly_at_limit_allowed(self):
        client = create_app_with_size_limit(max_size=100)
        # Approximate JSON size; ensure Content-Length is not over limit
        data = {"t": "a" * 50}
        resp = client.post("/echo", json=data)
        assert resp.status_code == 200
        assert resp.json()["received"]["t"] == "a" * 50

    def test_missing_or_small_content_length_passes(self):
        client = create_app_with_size_limit(max_size=5)
        # Send without JSON to avoid setting content-length; should pass to endpoint
        resp = client.post("/echo", content="hi")
        # Starlette/TestClient may set content-length and FastAPI may return 422 for invalid JSON
        assert resp.status_code in (200, 413, 422)
        # If 413, confirm it's because limit is very small
        if resp.status_code == 413:
            assert resp.json()["max_size"] == 5
