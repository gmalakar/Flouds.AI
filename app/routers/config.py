# =============================================================================
# File: config.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: config.py
# Copied from FloudsVector canonical implementation
# =============================================================================
# typing imports not required here

from fastapi import APIRouter, Body, Header, HTTPException, Query, Request, status

from app.logger import get_logger
from app.models.config_request import ConfigRequest
from app.models.delete_config_request import DeleteConfigRequest
from app.modules.key_manager import key_manager
from app.services.config_service import config_service
from app.utils.common_utils import CommonUtils

logger = get_logger("config.router")

router = APIRouter()

# Module-level parameter defaults to avoid function-call defaults (flake8 B008)
TENANT_HEADER = Header("", alias="X-Tenant-Code")
KEY_QUERY = Query(..., description="Config key (part of composite PK)")
TENANT_QUERY = Query(
    "",
    description="Tenant code (part of composite PK). Empty string for default tenant",
)

# Move Body defaults to module-level constants to avoid function-call defaults
ADD_CONFIG_PAYLOAD = Body(
    ...,
    examples=[
        {
            "summary": "Example payload",
            "value": {
                "key": "cors_origins",
                "value": '["https://example.com"]',
                "tenant_code": "",
                "encrypted": False,
            },
        }
    ],
)
UPDATE_CONFIG_PAYLOAD = Body(...)
DELETE_CONFIG_PAYLOAD = Body(...)


# Router-level startup handlers are deprecated. DB initialization happens
# in the application's lifespan handler (`app/main.py`) so this router no
# longer needs its own startup event.


@router.post("/add")
def add_config(
    request: Request,
    payload: ConfigRequest = ADD_CONFIG_PAYLOAD,
    tenant_code: str = TENANT_HEADER,
) -> dict:
    """Create a new config entry. Admins only.

    Payload fields:
      - key: string
      - value: string (for structured data, JSON-encode before sending)
      - tenant_code: optional (overrides header)
      - encrypted: optional boolean (default False)
    """
    # Validate tenant match and compute effective tenant
    client_id = getattr(request.state, "client_id", None)
    CommonUtils.validate_tenant_match(client_id, tenant_code, payload.tenant_code)
    tcode = payload.tenant_code or tenant_code or request.state.tenant_code or ""
    # Only superadmin may operate on global (empty) tenant scope. For tenant-scoped
    # operations require admin (or superadmin).
    if tcode == "":
        if not client_id or not key_manager.is_super_admin(client_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Superadmin access required for global config",
            )
    else:
        if not client_id or not key_manager.is_admin(client_id, tcode):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
            )

    key = payload.key
    value = payload.value
    enc = bool(payload.encrypted)

    if not key:
        raise HTTPException(status_code=400, detail="Missing key in payload")

    try:
        config_service.set_config(key, str(value), tenant_code=tcode, encrypted=enc)
        config_service.load_and_apply_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"success": True, "key": key, "tenant_code": tcode}


@router.get("/get")
def get_config(
    request: Request,
    key: str = KEY_QUERY,
    tenant_code: str = TENANT_QUERY,
    tenant_header: str = TENANT_HEADER,
) -> dict:
    """Retrieve a config value by composite primary key (key + tenant_code).
    If the stored value is encrypted, return a masked response (do not return decrypted or ciphertext).
    """
    # Validate tenant header vs query parameter when both present
    client_id = getattr(request.state, "client_id", None)
    CommonUtils.validate_tenant_match(client_id, tenant_header, tenant_code)
    tcode = tenant_code or tenant_header or ""
    try:
        val, encrypted = config_service.get_config_meta(key, tenant_code=tcode)
        if val is None and not encrypted:
            # not found
            raise HTTPException(status_code=404, detail="Key not found")
        if encrypted:
            # Mask encrypted values instead of returning decrypted or ciphertext
            return {
                "key": key,
                "value": "<encrypted>",
                "encrypted": True,
                "tenant_code": tcode,
            }
        return {"key": key, "value": val, "encrypted": False, "tenant_code": tcode}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/update")
def update_config(
    request: Request,
    payload: ConfigRequest = UPDATE_CONFIG_PAYLOAD,
    tenant_code: str = TENANT_HEADER,
) -> dict:
    """Update an existing config entry. Admins only.

    Payload must include `key` and `tenant_code` (composite PK), plus `value` and optional `encrypted`.
    """
    # Validate tenant header vs payload and compute effective tenant
    client_id = getattr(request.state, "client_id", None)
    CommonUtils.validate_tenant_match(client_id, tenant_code, payload.tenant_code)
    key = payload.key
    tcode = (
        payload.tenant_code
        if payload.tenant_code is not None
        else tenant_code or request.state.tenant_code or ""
    )
    value = payload.value
    enc = bool(payload.encrypted)

    if not key:
        raise HTTPException(status_code=400, detail="Missing key in payload")
    if tcode is None:
        raise HTTPException(status_code=400, detail="Missing tenant_code in payload")

    # Only superadmin may operate on global (empty) tenant scope. For tenant-scoped
    # operations require admin (or superadmin).
    if tcode == "":
        if not client_id or not key_manager.is_super_admin(client_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Superadmin access required for global config",
            )
    else:
        if not client_id or not key_manager.is_admin(client_id, tcode):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
            )

    try:
        config_service.set_config(key, str(value), tenant_code=tcode, encrypted=enc)
        config_service.load_and_apply_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"success": True, "key": key, "tenant_code": tcode}


@router.delete("/delete")
def delete_config(
    request: Request,
    payload: DeleteConfigRequest = DELETE_CONFIG_PAYLOAD,
    tenant_code: str = TENANT_HEADER,
) -> dict:
    """Delete a config entry by composite PK. Payload must include `key` and `tenant_code`.
    Admins only.
    """
    client_id = getattr(request.state, "client_id", None)
    CommonUtils.validate_tenant_match(client_id, tenant_code, payload.tenant_code)
    key = payload.key
    tcode = (
        payload.tenant_code
        if payload.tenant_code is not None
        else tenant_code or request.state.tenant_code or ""
    )

    if not key:
        raise HTTPException(status_code=400, detail="Missing key in payload")
    if tcode is None:
        raise HTTPException(status_code=400, detail="Missing tenant_code in payload")

    # Only superadmin may operate on global (empty) tenant scope. For tenant-scoped
    # operations require admin (or superadmin).
    if tcode == "":
        if not client_id or not key_manager.is_super_admin(client_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Superadmin access required for global config",
            )
    else:
        if not client_id or not key_manager.is_admin(client_id, tcode):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
            )

    try:
        config_service.delete_config(key, tenant_code=tcode)
        config_service.load_and_apply_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"success": True, "key": key, "tenant_code": tcode}
