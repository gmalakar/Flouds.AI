# =============================================================================
# File: input_validator.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: input_validator.py
# Small helper to validate tenant codes used by request models
# =============================================================================
import re

_TENANT_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def validate_tenant_code(v: str) -> str:
    if v is None:
        raise ValueError("tenant_code is required")
    s = str(v)
    s = s.strip()
    if not (1 <= len(s) <= 128):
        raise ValueError("tenant_code length must be between 1 and 128 characters")
    if not _TENANT_RE.match(s):
        raise ValueError("tenant_code contains invalid characters")
    return s
