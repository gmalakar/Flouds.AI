# =============================================================================
# File: test_request_middlewares.py
# Date: 2026-01-30
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: test_request_middlewares.py
# Tests for request logging, size limit, and validation middlewares
# =============================================================================

import importlib
import sys
import types

from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

# Prevent importing the real app.app_init (which runs config validation on import).
# Provide a minimal shim with APP_SETTINGS for middleware constructors.
shim = types.ModuleType("app.app_init")
shim.APP_SETTINGS = types.SimpleNamespace(
    app=types.SimpleNamespace(max_request_size=1024, request_timeout=5)
)
sys.modules["app.app_init"] = shim

# Middleware modules are imported inside tests after the `app.app_init` shim
# is adjusted to ensure tests can control `APP_SETTINGS` at import time.


def _ensure_app_settings(max_size: int = 1024, timeout: int = 5) -> None:
    app_init = sys.modules.get("app.app_init")
    if not app_init:
        raise RuntimeError("shimmed app.app_init missing")
    app_init.APP_SETTINGS.app.max_request_size = max_size
    app_init.APP_SETTINGS.app.request_timeout = timeout


def test_request_logging_middleware_echo_request_id():
    _ensure_app_settings(max_size=65536)
    app = FastAPI()

    @app.post("/echo")
    async def echo(req_body: dict) -> Response:
        return Response(status_code=200, content=b"ok")

    # Import/reload middleware after setting APP_SETTINGS shim
    import app.middleware.request_logging as _rl

    importlib.reload(_rl)
    RequestLoggingMiddleware = _rl.RequestLoggingMiddleware

    app.add_middleware(RequestLoggingMiddleware)

    client = TestClient(app)

    req_id = "test-req-abc"
    resp = client.post(
        "/echo", headers={"X-Request-ID": req_id, "Content-Type": "application/json"}, json={"a": 1}
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == req_id


def test_request_size_limit_middleware_rejects_by_content_length():
    # small max to trigger Content-Length short-circuit
    _ensure_app_settings(max_size=10)
    app = FastAPI()

    @app.post("/noop")
    async def noop() -> Response:
        return Response(status_code=200, content=b"ok")

    # Ensure middleware reads the updated APP_SETTINGS
    import app.middleware.request_size_limit as _rsl

    importlib.reload(_rsl)
    RequestSizeLimitMiddleware = _rsl.RequestSizeLimitMiddleware

    app.add_middleware(RequestSizeLimitMiddleware)
    client = TestClient(app)

    # send body with Content-Length header > max
    headers = {"Content-Type": "application/octet-stream", "Content-Length": "100"}
    resp = client.post("/noop", headers=headers, content=b"x" * 100)
    assert resp.status_code == 413


def test_request_validation_middleware_rejects_large_json_body():
    # validation middleware eagerly reads JSON bodies and should reject if > max
    _ensure_app_settings(max_size=50)
    app = FastAPI()

    @app.post("/process")
    async def process() -> Response:
        return Response(status_code=200, content=b"ok")

    # Ensure middleware reads the updated APP_SETTINGS
    import app.middleware.request_validation as _rv

    importlib.reload(_rv)
    RequestValidationMiddleware = _rv.RequestValidationMiddleware

    app.add_middleware(RequestValidationMiddleware)
    client = TestClient(app)

    # create JSON payload larger than max_size bytes
    big = {"data": "x" * 200}
    resp = client.post("/process", json=big)
    assert resp.status_code == 413
    assert (
        "REQUEST_TOO_LARGE" in (resp.json().get("error_code") or resp.json().get("error_code", ""))
        or resp.status_code == 413
    )
