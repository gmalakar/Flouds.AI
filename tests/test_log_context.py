# =============================================================================
# File: test_log_context.py
# Date: 2026-01-17
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: test_log_context.py
# Date: 2026-01-17
# =============================================================================

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.log_context import LogContextMiddleware


def test_request_id_header_echo():
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    app.add_middleware(LogContextMiddleware)
    client = TestClient(app)

    # Provide a request id and expect to get it back
    req_id = "test-req-123"
    resp = client.get("/ping", headers={"X-Request-ID": req_id})
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == req_id
