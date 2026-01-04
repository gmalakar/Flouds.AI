# =============================================================================
# File: embedding_response.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Any, Dict, List, Optional, cast

from pydantic import Field

from app.models.base_response import BaseResponse
from app.models.embedded_chunk import EmbededChunk


class EmbeddingResponse(BaseResponse):
    """
    Response model for single text embedding.
    """

    results: List[EmbededChunk] = Field(
        cast(Any, ...), description="Embedding chunks for the input text."
    )
    used_parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Actual parameters used for embedding generation.",
    )


class EmbeddingBatchResponse(BaseResponse):
    """
    Response model for batch text embedding.
    """

    results: List[EmbededChunk] = Field(
        cast(Any, ...), description="Embedding chunks for all input texts in the batch."
    )
    used_parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Actual parameters used for embedding generation.",
    )
