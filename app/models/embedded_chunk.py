# =============================================================================
# File: embedded_chunk.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from pydantic import BaseModel, Field


class EmbededChunk(BaseModel):
    vector: list[float] = Field(
        ..., description="The generated embedding for the text chunk."
    )
    chunk: str = Field(..., description="The original text chunk that was embedded.")

    joined_chunk: bool = Field(
        False,
        description="Indicates whether the chunk is part of a joined text chunk.",
    )
    only_vector: bool = Field(
        False,
        description="Indicates whether only the vector representation is available.",
    )
