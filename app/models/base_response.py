# =============================================================================
# File: base_response.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """
    Base response model for API responses.
    """

    success: bool = Field(
        True, description="Indicates whether the operation was successful."
    )
    message: str = Field(
        "Operation completed successfully.",
        description="A message providing additional information about the operation.",
    )
    model: str = Field(
        "none",
        description="The model used for generating the response.",
    )
    time_taken: float = Field(
        0.0, description="The time taken to complete the operation in seconds."
    )
