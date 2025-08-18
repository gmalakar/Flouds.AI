# =============================================================================
# File: embedding_request.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Annotated

from pydantic import Field

from app.models.base_request import BaseRequest


class EmbeddingBaseRequest(BaseRequest):
    projected_dimension: int = Field(
        None,
        gt=0,
        description="Target embedding dimension for projection. Overrides model config if provided.",
    )
    join_chunks: bool = Field(
        False,
        description="Whether to join the chunks of the embedding into a single string. Defaults to False.",
    )
    join_by_pooling_strategy: str = Field(
        None,
        description="The string used to join the chunks if join_chunks is True. Defaults to a single space.",
    )
    output_large_text_upon_join: bool = Field(
        False,
        description="Whether to output the large text upon joining. Defaults to False.",
    )

    pooling_strategy: str = Field(
        None,
        description="Pooling strategy: 'mean', 'max', 'cls', 'first', 'last', 'none'. Overrides model config if provided.",
    )

    max_length: int = Field(
        None,
        gt=0,
        description="Maximum token length for input text. Overrides model config if provided.",
    )

    chunk_logic: str = Field(
        None,
        description="Text chunking strategy: 'sentence', 'paragraph', 'fixed'. Overrides model config if provided.",
    )

    chunk_overlap: int = Field(
        None,
        ge=0,
        description="Number of overlapping tokens/sentences between chunks. Overrides model config if provided.",
    )

    chunk_size: int = Field(
        None,
        gt=0,
        description="Fixed chunk size in tokens (used with 'fixed' chunk_logic). Overrides model config if provided.",
    )

    legacy_tokenizer: bool = Field(
        None,
        description="Use legacy tokenizer for compatibility with older models. Overrides model config if provided.",
    )

    normalize: bool = Field(
        None,
        description="Normalize embedding vectors to unit length. Overrides model config if provided.",
    )

    force_pooling: bool = Field(
        None,
        description="Force pooling even for single-token sequences. Overrides model config if provided.",
    )

    lowercase: bool = Field(
        None,
        description="Convert input text to lowercase before processing. Overrides model config if provided.",
    )

    remove_emojis: bool = Field(
        None,
        description="Remove emojis and non-ASCII characters from input text. Overrides model config if provided.",
    )

    use_optimized: bool = Field(
        None,
        description="Use optimized ONNX model if available. Overrides model config if provided.",
    )


class EmbeddingRequest(EmbeddingBaseRequest):
    """
    Request model for text embedding.
    """

    input: str = Field(
        ..., min_length=1, description="The input text to be embedded. Cannot be empty."
    )


class EmbeddingBatchRequest(EmbeddingBaseRequest):
    """
    Request model for batch text embedding.
    """

    inputs: list[Annotated[str, Field(min_length=1)]] = Field(
        ...,
        min_length=1,
        description="The input texts to be embedded. Must contain at least one non-empty text.",
    )
