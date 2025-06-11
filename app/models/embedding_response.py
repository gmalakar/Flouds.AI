# =============================================================================
# File: embedding_response.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import List

from pydantic import Field

from app.models.base_response import BaseResponse
from app.models.embedded_chunk import EmbededChunk


class EmbeddingResponse(BaseResponse):
    """
    Response model for text embedding.
    """

    results: List[EmbededChunk] = Field(
        ..., description="A list of embedding chunks for the input texts."
    )


class EmbeddingBatchResponse(BaseResponse):
    """
    Response model for batch text embedding.
    """

    results: List[EmbededChunk] = Field(
        ..., description="A list of embedding responses for the input texts."
    )
