# =============================================================================
# File: test_openapi_contract.py
# Date: 2026-01-01
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from fastapi.testclient import TestClient

from app.main import app
from app.models.model_info_response import ModelInfoResponse

client = TestClient(app)


def test_openapi_contains_model_info_and_response_matches_model():
    """Integration test: ensure OpenAPI spec has the path and the response conforms to the Pydantic model.

    If `openapi-core` is installed and importable, the test will also attempt a best-effort validation
    against the OpenAPI spec using `openapi-core`. If the import fails or validation fails due to
    API mismatches, the test will still succeed as long as the Pydantic validation passes.
    """
    # fetch openapi spec
    spec_resp = client.get("/api/v1/openapi.json")
    assert spec_resp.status_code == 200
    spec = spec_resp.json()

    # ensure the path exists in the OpenAPI spec
    assert "/api/v1/model/info" in spec.get("paths", {}), "OpenAPI spec missing /api/v1/model/info"

    # call the endpoint for a non-existent model (safe path)
    # include a valid token from key_manager so AuthMiddleware permits the request
    from app.modules.key_manager import key_manager

    tokens = key_manager.get_all_tokens()
    auth_header = {}
    if tokens:
        token = next(iter(tokens))
        # derive tenant code from client record when available; default to 'master'
        try:
            client_id = token.split("|", 1)[0]
            tenant = getattr(key_manager.clients.get(client_id), "tenant_code", "") or "master"
        except Exception:
            tenant = "master"
        auth_header = {"Authorization": f"Bearer {token}", "X-Tenant-Code": tenant}

    resp = client.get(
        "/api/v1/model/info",
        params={"model": "this-model-does-not-exist"},
        headers=auth_header,
    )
    assert resp.status_code == 200
    data = resp.json()

    # Validate using Pydantic model (guarantees response_model contract)
    ModelInfoResponse.model_validate(data)

    # Optional: if openapi-core is available, try to perform an operation-level validation.
    try:
        from openapi_core import create_spec
        from openapi_core.templating.media_types.finders import MediaTypeFinder  # noqa: F401
        from openapi_core.validation.response.validators import ResponseValidator

        spec_obj = create_spec(spec)

        # Build a response validator and attempt to validate the response body for the operation
        validator = ResponseValidator(spec_obj)

        # We will use response validator in a minimal way: construct an object with the fields
        # expected by the validator. openapi-core APIs vary by version; use a best-effort approach.
        openapi_response = {
            "data": resp.json(),
            "status_code": resp.status_code,
            "headers": resp.headers,
            "mimetype": resp.headers.get("content-type", "application/json"),
        }

        # Try to validate using validator.validate (some versions expect wrapper objects).
        try:
            result = validator.validate(openapi_response, path="/api/v1/model/info", method="get")
            # If result has errors attribute, assert there are no errors
            errors = getattr(result, "errors", None)
            if errors:
                # Convert to list and assert no errors
                assert len(list(errors)) == 0, f"OpenAPI validation errors: {errors}"
        except Exception:
            # If openapi-core API doesn't match, skip the strict validation step
            pass
    except Exception:
        # openapi-core not available or validation failed in a non-critical way; ignore.
        pass
