# =============================================================================
# File: base_request.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from pydantic import BaseModel, Field


class BaseRequest(BaseModel):
    """
    Request model for text summarization.
    """

    model: str = Field(
        ...,
        min_length=1,
        description="The model name to use. This field is required and cannot be empty.",
    )
