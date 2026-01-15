# =============================================================================
# File: common_utils.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: common_utils.py
# Small common utilities used by routers
# =============================================================================
from typing import Optional

from fastapi import HTTPException, status


class CommonUtils:
    @staticmethod
    def validate_tenant_match(
        client_id: Optional[str],
        header_tenant: Optional[str],
        payload_tenant: Optional[str],
    ) -> None:
        """Validate that tenant identifiers provided in header and payload do not conflict.

        This helper intentionally keeps logic minimal: if both header and payload
        tenant codes are present and different, raise a 400. Tenant authorization
        (admin/superadmin) is handled by KeyManager checks elsewhere.
        """
        if header_tenant and payload_tenant and header_tenant != payload_tenant:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Conflicting tenant codes in header and payload",
            )
