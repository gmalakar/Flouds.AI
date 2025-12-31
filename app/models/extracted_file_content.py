# =============================================================================
# File: extracted_file_content.py
# Date: 2025-12-21
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from pydantic import BaseModel, Field


class ExtractedFileContent(BaseModel):
    content: str = Field(..., description="The extracted file content as text.")

    item_number: int = Field(
        0,
        description="The item number (page, paragraph, row, etc.) from which the content was extracted.",
    )

    content_as: str = Field(
        "text", description="The format in which the content was extracted."
    )
