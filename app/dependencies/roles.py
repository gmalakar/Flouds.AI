# =============================================================================
# File: roles.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""
Simple role dependency helpers (example).

This mirrors the generator in FloudsVector.Py and is useful for config admin
examples and tests. Replace header-based checks with real token validation
in production.
"""

from typing import Optional

from fastapi import Header, HTTPException, status

# Module-level Header defaults to avoid function-call defaults (flake8 B008)
X_USER_ID_HEADER = Header(None, alias="X-User-Id")
X_USER_ROLE_HEADER = Header(None, alias="X-User-Role")
X_TENANT_CODE_HEADER = Header(None, alias="X-Tenant-Code")


class UserContext:
    def __init__(self, user_id: Optional[str], role: str, tenant: Optional[str]):
        self.user_id = user_id
        self.role = role
        self.tenant = tenant


def get_user_context(
    x_user_id: Optional[str] = X_USER_ID_HEADER,
    x_user_role: Optional[str] = X_USER_ROLE_HEADER,
    x_tenant_code: Optional[str] = X_TENANT_CODE_HEADER,
) -> UserContext:
    role = (x_user_role or "user").lower()
    if role not in ("superadmin", "admin", "tenant-admin", "user"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role header")
    return UserContext(user_id=x_user_id, role=role, tenant=x_tenant_code)


def require_role(user_ctx: UserContext, allowed: list[str]) -> None:
    if user_ctx.role not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
