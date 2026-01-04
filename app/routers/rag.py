# =============================================================================
# File: rag.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from fastapi import APIRouter, HTTPException

from app.exceptions import FloudsBaseException
from app.logger import get_logger
from app.models.prompt_request import PromptRequest
from app.models.prompt_response import PromptResponse
from app.models.rag_request import RAGRequest
from app.services.prompt_service import PromptProcessor
from app.utils.error_handler import ErrorHandler

router = APIRouter()
logger = get_logger("rag_router")


@router.post("/generate", response_model=PromptResponse)
async def generate_answer(request: RAGRequest) -> PromptResponse:
    """Generate answer using RAG (Retrieval-Augmented Generation)."""
    logger.debug(
        f"RAG request for model: {request.model}, query: {request.query[:100]}"
    )

    try:
        # Format prompt for T5/summarization model
        prompt = f"{request.instruction}\nQuestion: {request.query}\nContext: {request.context}\nAnswer:"

        # Convert to prompt request, carry tenant context through
        prompt_request = PromptRequest(
            model=request.model,
            input=prompt,
            temperature=request.temperature,
            tenant_code=request.tenant_code,
        )

        # Use existing prompt processor service
        response = PromptProcessor.summarize(prompt_request)
        response.message = "RAG answer generated successfully"

        return response

    except FloudsBaseException as e:
        status_code = ErrorHandler.get_http_status(e)
        raise HTTPException(status_code=status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in RAG endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")
