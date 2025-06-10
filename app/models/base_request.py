# =============================================================================
# File: base_request.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from pydantic import BaseModel, Field, condecimal


class BaseRequest(BaseModel):
    """
    Request model for text summarization.
    """

    model: str = Field(
        ...,
        description="The model name to use for summarization. This field is required.",
    )
