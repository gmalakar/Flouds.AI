# =============================================================================
# File: rag_request.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Optional

from pydantic import Field

from app.models.base_request import BaseRequest


class RAGRequest(BaseRequest):
    """Request model for RAG (Retrieval-Augmented Generation)."""

    query: str = Field(..., description="The question or query to answer")
    context: str = Field(
        ..., description="Retrieved context/chunks for answering the query"
    )
    instruction: Optional[str] = Field(
        default="Answer the question based on the context below:",
        description="Instruction for how to process the query and context",
    )
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
