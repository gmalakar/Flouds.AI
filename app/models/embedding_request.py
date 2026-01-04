# =============================================================================
# File: embedding_request.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import List, Optional, cast

from pydantic import Field

from app.models.base_request import BaseRequest


class EmbeddingBaseRequest(BaseRequest):
    projected_dimension: Optional[int] = Field(
        default=None,
        gt=0,
        description="Target embedding dimension for projection. Overrides model config if provided.",
    )
    join_chunks: Optional[bool] = Field(
        default=None,
        description="Whether to join the chunks of the embedding into a single string. Overrides model config if provided.",
    )
    join_by_pooling_strategy: Optional[str] = Field(
        default=None,
        description="The string used to join the chunks if join_chunks is True. Overrides model config if provided.",
    )
    output_large_text_upon_join: Optional[bool] = Field(
        default=None,
        description="Whether to output the large text upon joining. Overrides model config if provided.",
    )

    pooling_strategy: Optional[str] = Field(
        default=None,
        description="Pooling strategy: 'mean', 'max', 'cls', 'first', 'last', 'none'. Overrides model config if provided.",
    )

    max_length: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum token length for input text. Overrides model config if provided.",
    )

    chunk_logic: Optional[str] = Field(
        default=None,
        description="Text chunking strategy: 'sentence', 'paragraph', 'fixed'. Overrides model config if provided.",
    )

    chunk_overlap: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of overlapping tokens/sentences between chunks. Overrides model config if provided.",
    )

    chunk_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="Fixed chunk size in tokens (used with 'fixed' chunk_logic). Overrides model config if provided.",
    )

    legacy_tokenizer: Optional[bool] = Field(
        default=None,
        description="Use legacy tokenizer for compatibility with older models. Overrides model config if provided.",
    )

    normalize: Optional[bool] = Field(
        default=None,
        description="Normalize embedding vectors to unit length. Overrides model config if provided.",
    )

    force_pooling: Optional[bool] = Field(
        default=None,
        description="Force pooling even for single-token sequences. Overrides model config if provided.",
    )

    lowercase: Optional[bool] = Field(
        default=None,
        description="Convert input text to lowercase before processing. Overrides model config if provided.",
    )

    remove_emojis: Optional[bool] = Field(
        default=None,
        description="Remove emojis and non-ASCII characters from input text. Overrides model config if provided.",
    )

    use_optimized: Optional[bool] = Field(
        default=None,
        description="Use optimized ONNX model if available. Overrides model config if provided.",
    )


class EmbeddingRequest(EmbeddingBaseRequest):
    """
    Request model for text embedding.
    """

    model: str = Field(
        default=cast(str, ...),
        min_length=1,
        description="The model name or identifier to use for embedding.",
    )

    input: str = Field(
        default=cast(str, ...),
        min_length=1,
        description="The input text to be embedded. Cannot be empty.",
    )


class EmbeddingBatchRequest(EmbeddingBaseRequest):
    """
    Request model for batch text embedding.
    """

    model: str = Field(
        default=cast(str, ...),
        min_length=1,
        description="The model name or identifier to use for embedding.",
    )

    inputs: List[str] = Field(
        default=cast(List[str], ...),
        min_length=1,
        description="The input texts to be embedded. Must contain at least one non-empty text.",
    )
