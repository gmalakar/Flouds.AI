# =============================================================================
# File: config_request.py
# Canonical request model for config operations
# =============================================================================
import re
from typing import Any, Optional, cast

from pydantic import Field, field_validator

from app.models.base_request import BaseRequest


class ConfigRequest(BaseRequest):
    """Request model for config operations.

    Fields:
      - key: config key (composite PK includes tenant_code)
      - value: config value (string; JSON-encode structured data before sending)
      - tenant_code: optional tenant code (inherited from BaseRequest)
      - encrypted: optional flag indicating value is stored encrypted
    """

    key: str = Field(cast(Any, ...), description="The config key.")  # type: ignore
    value: str = Field(cast(Any, ...), description="The config value.")  # type: ignore
    encrypted: Optional[bool] = Field(False, description="Is the value encrypted?")

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: Optional[str]) -> str:
        if v is None:
            raise ValueError("Missing key")
        if not (1 <= len(v) <= 128):
            raise ValueError("key must be between 1 and 128 characters long")
        if any(ch.isspace() for ch in v):
            raise ValueError("key must not contain whitespace characters")
        if not re.match(r"^[A-Za-z0-9_.:-]+$", v):
            raise ValueError(
                "key contains invalid characters; allowed are letters, digits, underscore (_), dot (.), colon (:), and hyphen (-)"
            )
        return v

    @field_validator("value")
    @classmethod
    def validate_value_length(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        max_len = 65536
        if len(v) > max_len:
            raise ValueError(f"value must be at most {max_len} characters long")
        return v
