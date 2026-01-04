# =============================================================================
# File: extracted_response.py
# Date: 2025-12-21
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
from typing import Any, List, cast

from pydantic import BaseModel, Field

from app.models.extracted_file_content import ExtractedFileContent


class ExtractedResponse(BaseModel):
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
    results: List[ExtractedFileContent] = Field(
        cast(Any, ...), description="A list of extracted file contents."
    )
    time_taken: float = Field(
        0.0, description="The time taken to complete the operation in seconds."
    )
