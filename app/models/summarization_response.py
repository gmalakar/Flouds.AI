# =============================================================================
# File: summarization_response.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from pydantic import BaseModel, Field

from app.models.base_response import BaseResponse


class SummarizationResponse(BaseResponse):
    """
    Response model for text summarization.
    """

    results: list[str] = Field(
        ..., description="The generated summary and related metadata as an object."
    )


# This class extends BaseResponse to include a results field, which is a list of strings containing the summarized text and related metadata.
# The Field decorator is used to provide additional metadata for the results field, such as a description.
