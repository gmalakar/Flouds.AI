# =============================================================================
# File: config_request_validators.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""
Example Pydantic model validators for configuration API requests.

This example mirrors the generator in FloudsVector.Py and provides strict
validators for config keys and values.
"""

import re
from typing import Any, Optional, cast

from pydantic import BaseModel, Field, field_validator

KEY_REGEX = re.compile(r"^[A-Za-z0-9_.:-]+$")


class ConfigRequest(BaseModel):
    key: str = Field(cast(Any, ...), min_length=1, max_length=128)
    value: Optional[str] = Field(cast(Any, ...), max_length=65536)

    @field_validator("key")
    def key_must_be_valid(cls, v: str) -> str:
        if any(ch.isspace() for ch in v):
            raise ValueError("key must not contain whitespace")
        if not KEY_REGEX.fullmatch(v):
            raise ValueError(
                "key contains invalid characters; allowed: A-Z a-z 0-9 _ . : -"
            )
        return v

    @field_validator("value")
    def value_length(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if len(v) > 65536:
            raise ValueError("value exceeds maximum size of 65536 characters")
        return v
