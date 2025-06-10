# =============================================================================
# File: summarization_request.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from pydantic import Field

from app.models.base_request import BaseRequest


class SummarizationBaseRequest(BaseRequest):
    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="The temperature to use for sampling. Must be between 0.0 and 2.0. Defaults to 0.0.",
    )


class SummarizationRequest(SummarizationBaseRequest):
    input: str = Field(..., description="The input text to be summarized.")


class SummarizationBatchRequest(SummarizationBaseRequest):
    inputs: list[str] = Field(..., description="The input texts to be summarized.")
