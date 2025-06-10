# =============================================================================
# File: embedding_request.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from pydantic import Field

from app.models.base_request import BaseRequest


class EmbeddingBaseRequest(BaseRequest):
    projected_dimension: int = Field(
        None,
        description="The dimension to which the embedding will be projected. Defaults to None.",
    )
    join_chunks: bool = Field(
        True,
        description="Whether to join the chunks of the embedding into a single string. Defaults to True.",
    )
    join_by_pooling_strategy: str = Field(
        None,
        description="The string used to join the chunks if join_chunks is True. Defaults to a single space.",
    )
    output_large_text_upon_join: bool = Field(
        False,
        description="Whether to output the large text upon joining. Defaults to False.",
    )


class EmbeddingRequest(EmbeddingBaseRequest):
    """
    Request model for text embedding.
    """

    input: str = Field(..., description="The input text to be embedded.")


class EmbeddingBatchRequest(EmbeddingBaseRequest):
    """
    Request model for batch text embedding.
    """

    inputs: list[str] = Field(..., description="The input texts to be embedded.")
