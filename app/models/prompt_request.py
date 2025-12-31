# =============================================================================
# File: prompt_request.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Annotated, Optional

from pydantic import Field

from app.models.base_request import BaseRequest


class PromptBaseRequest(BaseRequest):
    temperature: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="The temperature to use for sampling. Must be between 0.0 and 2.0. Defaults to 0.7.",
    )


class PromptRequest(PromptBaseRequest):
    input: str = Field(
        ...,
        min_length=1,
        description="The input text to be processed. Cannot be empty.",
    )


class PromptBatchRequest(PromptBaseRequest):
    inputs: list[Annotated[str, Field(min_length=1)]] = Field(
        ...,
        min_length=1,
        description="The input texts to be processed. Must contain at least one non-empty text.",
    )
