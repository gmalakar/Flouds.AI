# =============================================================================
# File: base_request.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: base_request.py
# Lightweight BaseRequest adapted from canonical implementation
# =============================================================================
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.requests import Request

from app.utils.input_validator import validate_tenant_code


class BaseRequest(BaseModel):
    """
    Base request model that includes optional `tenant_code` handling.
    """

    tenant_code: Optional[str] = Field(
        None,
        description="The tenant for which the request is made. If omitted, it will be resolved from the incoming request headers.",
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("tenant_code")
    @classmethod
    def validate_tenant_code_field(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return validate_tenant_code(v)

    def resolve_tenant(self, request: Request) -> str:
        if self.tenant_code:
            return self.tenant_code

        header = request.headers.get("X-Tenant-Code")
        if not header:
            header = getattr(request.state, "tenant_code", None)

        if not header:
            raise ValueError("Missing tenant code in request headers")

        validated = validate_tenant_code(header)
        self.tenant_code = validated
        return validated
        # end of canonical BaseRequest
