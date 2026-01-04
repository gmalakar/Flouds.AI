# =============================================================================
# File: prompt_response.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Any, cast

from pydantic import Field

from app.models.base_response import BaseResponse


class PromptResponse(BaseResponse):
    """
    Response model for prompt processing.
    """

    results: list[str] = Field(
        cast(Any, ...),
        description="The generated text results and related metadata as an object.",
    )


# This class extends BaseResponse to include a results field, which is a list of strings containing the generated text and related metadata.
# The Field decorator is used to provide additional metadata for the results field, such as a description.
