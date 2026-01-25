# =============================================================================
# File: test_auth_middleware.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient


def _make_app():
    app = FastAPI()

    @app.get("/api/v1/health")
    def health():
        return JSONResponse({"ok": True})

    @app.get("/private")
    def private():
        return JSONResponse({"secret": True})

    # add middleware under test
    from app.dependencies.auth import AuthMiddleware

    app.add_middleware(AuthMiddleware)
    return app


def test_public_endpoint_bypasses_auth(mock_app_settings):
    # Enable security but public endpoint should bypass auth checks
    mock_app_settings.security.enabled = True
    with (
        patch("app.modules.key_manager.key_manager.load_clients", return_value=None),
        patch(
            "app.modules.key_manager.key_manager.get_all_tokens", return_value=["tok"]
        ),
    ):
        app = _make_app()
        client = TestClient(app)
        r = client.get("/api/v1/health")
        assert r.status_code == 200


def test_private_requires_tenant_and_auth(mock_app_settings):
    mock_app_settings.security.enabled = True

    with (
        patch("app.modules.key_manager.key_manager.load_clients", return_value=None),
        patch(
            "app.modules.key_manager.key_manager.get_all_tokens", return_value=["tok"]
        ),
    ):
        app = _make_app()
        client = TestClient(app)

        # Missing tenant header -> 401
        r = client.get("/private")
        assert r.status_code == 401
        assert "Missing X-Tenant-Code" in r.json().get("message", "")

        # With tenant but missing Authorization -> 401
        r = client.get("/private", headers={"X-Tenant-Code": "t1"})
        assert r.status_code == 401
        assert "Missing Authorization" in r.json().get("message", "")

        # With tenant and Authorization but invalid token -> 401
        with patch(
            "app.modules.key_manager.key_manager.authenticate_client", return_value=None
        ):
            r = client.get(
                "/private",
                headers={"X-Tenant-Code": "t1", "Authorization": "Bearer badtoken"},
            )
            assert r.status_code == 401
            assert "Invalid API token" in r.json().get("message", "")

        # With tenant and valid token -> 200
        mock_client_obj = Mock()
        mock_client_obj.client_id = "cid"
        mock_client_obj.client_type = "api_user"
        with patch(
            "app.modules.key_manager.key_manager.authenticate_client",
            return_value=mock_client_obj,
        ):
            r = client.get(
                "/private",
                headers={"X-Tenant-Code": "t1", "Authorization": "Bearer good"},
            )
            assert r.status_code == 200
