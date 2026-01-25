# =============================================================================
# File: test_model_info_property.py
# Date: 2026-01-01
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _auth_header():
    # include a valid token from key_manager so AuthMiddleware permits the request
    try:
        from app.modules.key_manager import key_manager

        tokens = key_manager.get_all_tokens()
        if tokens:
            token = next(iter(tokens))
            # Derive tenant from client record when available; default to 'master'
            try:
                client_id = token.split("|", 1)[0]
                tenant = getattr(key_manager.clients.get(client_id), "tenant_code", "") or "master"
            except Exception:
                tenant = "master"
            return {"Authorization": f"Bearer {token}", "X-Tenant-Code": tenant}
    except Exception:
        pass
    return {}


def test_model_info_property_not_found_returns_null():
    """When requesting a specific property for a missing model, API returns a `property` object with null value."""
    resp = client.get(
        "/api/v1/model/info",
        params={
            "model": "this-model-does-not-exist",
            "property_name": "auto_detected_params",
        },
        headers=_auth_header(),
    )
    assert resp.status_code == 200
    data = resp.json()

    # The compact response should include `results` with property_name and property_value (null)
    assert "results" in data
    results = data["results"]
    assert results.get("property_name") == "auto_detected_params"
    assert results.get("property_value") is None

    # The overall success flag should reflect the service result (model not found -> success False)
    assert data.get("success") is False
    assert "not found" in data.get("message", "").lower()


def test_model_info_nested_property_not_found_returns_null():
    """Request a nested property using dot notation; missing nested value should be returned as None on the property value."""
    resp = client.get(
        "/api/v1/model/info",
        params={
            "model": "this-model-does-not-exist",
            "property_name": "auto_detected_params.dimension",
        },
        headers=_auth_header(),
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "results" in data
    results = data["results"]
    assert results.get("property_name") == "auto_detected_params.dimension"
    assert results.get("property_value") is None

    assert data.get("success") is False
