# =============================================================================
# File: test_tenant_security.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import tenant_security  # noqa: F401
from app.middleware.tenant_security import (
    TenantCorsMiddleware,
    TenantTrustedHostMiddleware,
    _is_allowed,
    _match_pattern,
)


def test_match_pattern_basic():
    assert _match_pattern("example.com", "example.com")
    assert _match_pattern("anything", "*")
    assert _match_pattern("api.example.com", "*.example.com")
    assert _match_pattern("example.com", "*.example.com")
    assert not _match_pattern("badexample.com", "*.example.com")
    # regex
    assert _match_pattern("sub.example.org", "re:^(?:.+\\.)?example\\.org$")


def test_is_allowed_list():
    allowed = ["*.example.com", "api.svc.local", "re:^test-\\d+\\.local$"]
    assert _is_allowed("example.com", allowed)
    assert _is_allowed("api.example.com", allowed)
    assert _is_allowed("api.svc.local", allowed)
    assert _is_allowed("test-123.local", allowed)
    assert not _is_allowed("evil.com", allowed)


@pytest.fixture
def client_with_middleware(monkeypatch):
    """Create a TestClient with the tenant host/CORS middleware applied.

    We monkeypatch `config_service` functions so the middleware gets deterministic
    allowed lists for the tests.
    """
    # Monkeypatch config_service functions used by middleware
    import app.services.config_service as config_service

    monkeypatch.setattr(
        config_service, "get_trusted_hosts", lambda tenant_code="": ["*.example.com"]
    )
    monkeypatch.setattr(
        config_service,
        "get_cors_origins",
        lambda tenant_code="": ["https://app.example.com"],
    )

    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    app.add_middleware(TenantTrustedHostMiddleware)
    app.add_middleware(TenantCorsMiddleware)

    return TestClient(app)


def test_trusted_host_middleware_allows_and_blocks(client_with_middleware):
    client = client_with_middleware
    headers = {"host": "example.com", "X-Tenant-Code": "t1"}
    r = client.get("/ping", headers=headers)
    assert r.status_code == 200

    headers = {"host": "api.example.com", "X-Tenant-Code": "t1"}
    r = client.get("/ping", headers=headers)
    assert r.status_code == 200

    headers = {"host": "evil.com", "X-Tenant-Code": "t1"}
    r = client.get("/ping", headers=headers)
    assert r.status_code == 403


def test_cors_middleware_allows_and_blocks_origins(client_with_middleware):
    client = client_with_middleware
    headers = {
        "host": "example.com",
        "X-Tenant-Code": "t1",
        "Origin": "https://app.example.com",
    }
    r = client.get("/ping", headers=headers)
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"

    headers = {
        "host": "example.com",
        "X-Tenant-Code": "t1",
        "Origin": "https://evil.com",
    }
    r = client.get("/ping", headers=headers)
    assert r.status_code == 403


def test_cors_preflight_returns_204(client_with_middleware):
    client = client_with_middleware
    headers = {
        "host": "example.com",
        "X-Tenant-Code": "t1",
        "Origin": "https://app.example.com",
    }
    r = client.options("/ping", headers=headers)
    assert r.status_code == 204
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"


def test_case_insensitive_host_matching(client_with_middleware):
    client = client_with_middleware
    # Uppercase host should still be allowed
    headers = {"host": "API.EXAMPLE.COM", "X-Tenant-Code": "t1"}
    r = client.get("/ping", headers=headers)
    assert r.status_code == 200


def test_wildcard_to_root_and_subdomains(client_with_middleware):
    client = client_with_middleware
    # Root domain
    headers = {"host": "example.com", "X-Tenant-Code": "t1"}
    r = client.get("/ping", headers=headers)
    assert r.status_code == 200

    # Direct subdomain
    headers = {"host": "api.example.com", "X-Tenant-Code": "t1"}
    r = client.get("/ping", headers=headers)
    assert r.status_code == 200

    # Deep subdomain
    headers = {"host": "deep.sub.api.example.com", "X-Tenant-Code": "t1"}
    r = client.get("/ping", headers=headers)
    assert r.status_code == 200
