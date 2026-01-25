# =============================================================================
# File: models.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Data models and constants for embedder service."""

from typing import List

from pydantic import BaseModel, Field

from app.models.embedded_chunk import EmbededChunk

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MAX_LENGTH = 256
DEFAULT_PROJECTED_DIMENSION = 128
DEFAULT_BATCH_SIZE = 50


# ============================================================================
# Result Models
# ============================================================================


class SingleEmbeddingResult(BaseModel):
    """Result model for single text embedding."""

    vector: List[float]
    message: str
    success: bool = Field(default=True)

    @property
    def embedding_results(self) -> List[float]:
        """Backward compatibility property."""
        return self.vector

    # Legacy property for existing tests
    EmbeddingResults = embedding_results


class ChunkEmbeddingResult(BaseModel):
    """Result model for chunked text embedding."""

    embedding_chunks: List[EmbededChunk]
    message: str
    success: bool = Field(default=True)
    used_parameters: dict = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    @property
    def embedding_results(self) -> List[EmbededChunk]:
        """Backward compatibility property."""
        return self.embedding_chunks

    # Legacy property for existing tests
    EmbeddingResults = embedding_results
