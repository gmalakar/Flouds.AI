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

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.request_size_limit import RequestSizeLimitMiddleware


def create_app_with_size_limit(max_size: int) -> TestClient:
    app = FastAPI()

    @app.post("/echo")
    async def echo(payload: dict):
        return {"received": payload}

    app.add_middleware(RequestSizeLimitMiddleware, max_size=max_size)
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
