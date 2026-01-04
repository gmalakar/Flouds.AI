# =============================================================================
# File: prompt_request.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import List, Optional, cast

from pydantic import Field

from app.models.base_request import BaseRequest


class PromptBaseRequest(BaseRequest):
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="The temperature to use for sampling. Must be between 0.0 and 2.0. Defaults to 0.7.",
    )


class PromptRequest(PromptBaseRequest):
    # Optional model name to select specific model/runtime. If not provided,
    # services will fall back to a sensible default.
    model: Optional[str] = Field(
        default=cast(Optional[str], None),
        description="Optional model name to use for processing the request.",
    )
    input: str = Field(
        default=cast(str, ...),
        min_length=1,
        description="The input text to be processed. Cannot be empty.",
    )


class PromptBatchRequest(PromptBaseRequest):
    # Optional model name for batch requests as well.
    model: Optional[str] = Field(
        default=cast(Optional[str], None),
        description="Optional model name to use for processing the batch request.",
    )
    inputs: List[str] = Field(
        default=cast(List[str], ...),
        min_length=1,
        description="The input texts to be processed. Must contain at least one non-empty text.",
    )
