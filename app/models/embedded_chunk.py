# =============================================================================
# File: embedded_chunk.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Any, Optional, cast

from pydantic import BaseModel, Field


class EmbededChunk(BaseModel):
    vector: list[float] = Field(
        cast(Any, ...), description="The generated embedding for the text chunk."
    )
    chunk: str = Field(
        cast(Any, ...), description="The original text chunk that was embedded."
    )

    joined_chunk: bool = Field(
        False,
        description="Indicates whether the chunk is part of a joined text chunk.",
    )
    only_vector: bool = Field(
        False,
        description="Indicates whether only the vector representation is available.",
    )

    item_number: Optional[int] = Field(
        None,
        description="The item number (page, paragraph, row, etc.) from which the content was extracted.",
    )
    content_as: Optional[str] = Field(
        None,
        description="The format in which the content was extracted.",
    )
