# =============================================================================
# File: embedder.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from app.logger import get_logger
from app.models.embedding_request import EmbeddingBatchRequest, EmbeddingRequest
from app.models.embedding_response import EmbeddingBatchResponse, EmbeddingResponse
from app.services.embedder_service import SentenceTransformer

router = APIRouter()
logger = get_logger("router")


@router.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest) -> EmbeddingResponse:
    logger.debug(f"Embedding request by model: {request.model}")
    response: EmbeddingResponse = SentenceTransformer.embed_text(request)
    return response


@router.post(
    "/embed_batch",
    response_model=EmbeddingBatchResponse,
)
async def embed_batch(requests: EmbeddingBatchRequest) -> EmbeddingBatchResponse:
    logger.debug(f"Embedding batch request, count: {len(requests.inputs)}")
    responses: EmbeddingBatchResponse = await SentenceTransformer.embed_batch_async(
        requests
    )
    return responses
