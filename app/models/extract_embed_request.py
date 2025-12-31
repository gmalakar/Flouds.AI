# =============================================================================
# File: extract_embed_request.py
# Date: 2025-12-22
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Union

from pydantic import Field

from app.models.embedding_request import EmbeddingBaseRequest


class ExtractEmbedRequest(EmbeddingBaseRequest):
    """
    Request model for combined file extraction and text embedding.
    Extracts text from a file and then generates embeddings.
    """

    file_content: Union[str, bytes] = Field(
        ...,
        description="File content to extract. Accepts base64 string or raw bytes.",
    )
    extention: str = Field(
        ...,
        min_length=1,
        description="The file extension. This field is required and cannot be empty.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_content": "SGVsbG8gd29ybGQ=",  # base64 for "Hello world"
                "extention": "txt",
                "model": "all-MiniLM-L6-v2",
            }
        }
