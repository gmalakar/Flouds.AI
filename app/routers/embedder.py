# =============================================================================
# File: embedder.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================


from fastapi import APIRouter, HTTPException

from app.exceptions import FloudsBaseException
from app.logger import get_logger
from app.models.embedding_request import EmbeddingBatchRequest, EmbeddingRequest
from app.models.embedding_response import EmbeddingBatchResponse, EmbeddingResponse
from app.services.embedder_service import SentenceTransformer
from app.utils.error_handler import ErrorHandler
from app.utils.log_sanitizer import sanitize_for_log

router = APIRouter()
logger = get_logger("router")


@router.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest) -> EmbeddingResponse:
    logger.debug("Embedding request by model: %s", sanitize_for_log(str(request.model)))
    try:
        response: EmbeddingResponse = SentenceTransformer.embed_text(request)
        return response
    except FloudsBaseException as e:
        status_code = ErrorHandler.get_http_status(e)
        raise HTTPException(status_code=status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in embedding endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/embed_batch",
    response_model=EmbeddingBatchResponse,
)
async def embed_batch(requests: EmbeddingBatchRequest) -> EmbeddingBatchResponse:
    logger.debug("Embedding batch request, count: %d", len(requests.inputs))
    try:
        responses: EmbeddingBatchResponse = await SentenceTransformer.embed_batch_async(
            requests
        )
        return responses
    except FloudsBaseException as e:
        status_code = ErrorHandler.get_http_status(e)
        raise HTTPException(status_code=status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in batch embedding endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")
