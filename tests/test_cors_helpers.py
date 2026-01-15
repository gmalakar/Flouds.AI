# =============================================================================
# File: test_cors_helpers.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import pytest
from starlette.responses import Response

from app.middleware.tenant_security import _apply_cors_headers, _cors_preflight


def test_cors_preflight_returns_204_and_headers():
    resp = _cors_preflight("https://app.example.com")
    assert resp.status_code == 204
    assert resp.headers["Access-Control-Allow-Origin"] == "https://app.example.com"
    assert resp.headers["Access-Control-Allow-Methods"] == "*"
    assert resp.headers["Access-Control-Allow-Headers"] == "*"
    assert resp.headers["Access-Control-Allow-Credentials"] == "true"


def test_cors_preflight_none_origin_uses_star():
    resp = _cors_preflight(None)
    assert resp.status_code == 204
    assert resp.headers["Access-Control-Allow-Origin"] == "*"


def test_apply_cors_headers_sets_headers_on_response():
    r = Response(content=b"ok", status_code=200)
    _apply_cors_headers(r, "https://app.example.com")
    assert r.status_code == 200
    assert r.headers["Access-Control-Allow-Origin"] == "https://app.example.com"
    assert r.headers["Access-Control-Allow-Methods"] == "*"
    assert r.headers["Access-Control-Allow-Headers"] == "*"
    assert r.headers["Access-Control-Allow-Credentials"] == "true"


def test_apply_cors_headers_none_origin_sets_star():
    r = Response(content=b"ok", status_code=200)
    _apply_cors_headers(r, None)
    assert r.headers["Access-Control-Allow-Origin"] == "*"
