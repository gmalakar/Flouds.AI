# =============================================================================
# File: test_tenant_security_integration.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.tenant_security import (
    TenantCorsMiddleware,
    TenantTrustedHostMiddleware,
)


class MockClient:
    def __init__(self, client_type: str = "api_user"):
        self.client_type = client_type
        self.client_id = "mock"


def make_app(monkeypatch, cors_origins=None, trusted_hosts=None, auth_return=None):
    # monkeypatch config_service
    import app.services.config_service as config_service

    monkeypatch.setattr(
        config_service,
        "get_cors_origins",
        lambda tenant_code="": (
            cors_origins if cors_origins is not None else ["https://app.example.com"]
        ),
    )
    monkeypatch.setattr(
        config_service,
        "get_trusted_hosts",
        lambda tenant_code="": (
            trusted_hosts if trusted_hosts is not None else ["*.example.com"]
        ),
    )

    # monkeypatch key_manager.authenticate_client on the module's singleton
    import app.modules.key_manager as km_module

    def _auth(token, tenant_code=""):
        return auth_return

    monkeypatch.setattr(km_module.key_manager, "authenticate_client", _auth)

    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    # Register CORS first as in production
    app.add_middleware(TenantCorsMiddleware)
    app.add_middleware(TenantTrustedHostMiddleware)

    return TestClient(app)


def test_trusted_host_fallback_allows_authenticated(monkeypatch):
    # Host trusted, origin blocked, client authenticated (non-superadmin) => allowed
    client = make_app(
        monkeypatch,
        cors_origins=["https://app.example.com"],
        trusted_hosts=["*.example.com"],
        auth_return=MockClient("api_user"),
    )

    headers = {
        "host": "example.com",
        "X-Tenant-Code": "t1",
        "Origin": "https://evil.com",
        "Authorization": "Bearer goodtoken",
    }
    r = client.get("/ping", headers=headers)
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://evil.com"


def test_trusted_host_fallback_blocks_unauthenticated(monkeypatch):
    # Host trusted, origin blocked, no auth => blocked
    client = make_app(
        monkeypatch,
        cors_origins=["https://app.example.com"],
        trusted_hosts=["*.example.com"],
        auth_return=None,
    )

    headers = {
        "host": "example.com",
        "X-Tenant-Code": "t1",
        "Origin": "https://evil.com",
    }
    r = client.get("/ping", headers=headers)
    assert r.status_code == 403


def test_superadmin_bypass_allows_even_when_host_not_trusted(monkeypatch):
    # Host not trusted, origin blocked, superadmin token => allowed
    client = make_app(
        monkeypatch,
        cors_origins=["https://app.example.com"],
        trusted_hosts=["different.com"],
        auth_return=MockClient("superadmin"),
    )

    headers = {
        "host": "example.com",
        "X-Tenant-Code": "t1",
        "Origin": "https://evil.com",
        "Authorization": "Bearer supertoken",
    }
    r = client.get("/ping", headers=headers)
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://evil.com"


def test_superadmin_bypass_allows_preflight(monkeypatch):
    client = make_app(
        monkeypatch,
        cors_origins=["https://app.example.com"],
        trusted_hosts=["different.com"],
        auth_return=MockClient("superadmin"),
    )

    headers = {
        "host": "example.com",
        "X-Tenant-Code": "t1",
        "Origin": "https://evil.com",
        "Authorization": "Bearer supertoken",
    }
    r = client.options("/ping", headers=headers)
    assert r.status_code == 204
    assert r.headers.get("access-control-allow-origin") == "https://evil.com"
