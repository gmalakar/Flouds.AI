# =============================================================================
# File: file_request.py
# Date: 2025-12-20
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Any, Union, cast

from pydantic import BaseModel, Field


class FileRequest(BaseModel):
    """Request model for file extraction.

    Supports either base64-encoded string or raw bytes for file_content.
    """

    file_content: Union[str, bytes] = Field(
        cast(Any, ...),
        description="File content to extract. Accepts base64 string or raw bytes.",
    )
    extention: str = Field(
        cast(Any, ...),
        min_length=1,
        description="The file extension. This field is required and cannot be empty.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_content": "SGVsbG8gd29ybGQ=",  # base64 for "Hello world"
                "extention": "txt",
            }
        }
