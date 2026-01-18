# =============================================================================
# File: model_metadata.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class ModelMetadata(BaseModel):
    """Typed, lightweight metadata for models used at runtime.

    This replaces free-form dicts previously stored in the lightweight
    metadata cache. The model intentionally uses permissive types for some
    fields to remain compatible with older callers.
    """

    resolved_path: Optional[str] = None
    model_folder_name: Optional[str] = None
    tasks: Optional[List[str]] = None
    timestamp: Optional[int] = None

    # Existence flags used to short-circuit filesystem checks. Values:
    #   - 'not_checked' (default)
    #   - 'true'
    #   - 'false'
    encoder_model_exists: str = "not_checked"
    decoder_model_exists: str = "not_checked"

    # Last time existence flags were checked (epoch seconds)
    last_checked: Optional[int] = None

    # Pydantic v2 configuration: allow extra fields and keep models mutable
    model_config = ConfigDict(extra="allow")
