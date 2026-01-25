# =============================================================================
# File: extract_embed_request.py
# Date: 2025-12-22
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Optional, Union, cast

from pydantic import ConfigDict, Field

from app.models.embedding_request import EmbeddingBaseRequest


class ExtractEmbedRequest(EmbeddingBaseRequest):
    """
    Request model for combined file extraction and text embedding.
    Extracts text from a file and then generates embeddings.
    """

    file_content: Union[str, bytes] = Field(
        default=cast(Union[str, bytes], ...),
        description="File content to extract. Accepts base64 string or raw bytes.",
    )
    extention: str = Field(
        default=cast(str, ...),
        min_length=1,
        description="The file extension. This field is required and cannot be empty.",
    )

    model: Optional[str] = Field(
        default=None,
        description="Optional model name or identifier to use for embedding when extracting files.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_content": "SGVsbG8gd29ybGQ=",  # base64 for "Hello world"
                "extention": "txt",
                "model": "all-MiniLM-L6-v2",
            }
        }
    )
