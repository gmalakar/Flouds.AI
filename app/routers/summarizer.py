# =============================================================================
# File: summarizer.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================


from fastapi import APIRouter, HTTPException

from app.exceptions import FloudsBaseException
from app.logger import get_logger
from app.models.summarization_request import (
    SummarizationBatchRequest,
    SummarizationRequest,
)
from app.models.summarization_response import SummarizationResponse
from app.services.summarizer_service import TextSummarizer
from app.utils.error_handler import ErrorHandler

router = APIRouter()
logger = get_logger("router")

# HINTS:
# - Both endpoints are async and call async methods from TextSummarizer.
# - The /summarize endpoint expects a SummarizationRequest and returns a SummarizationResponse.
# - The /summarize_batch endpoint expects a SummarizationBatchRequest and returns a list of SummarizationResponse.
# - Use type hints for FastAPI endpoint parameters and return types for better validation and editor support.


@router.post("/summarize", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest) -> SummarizationResponse:
    logger.debug(f"Summarization request by model: {request.model}")
    try:
        summary: SummarizationResponse = TextSummarizer.summarize(request)
        return summary
    except FloudsBaseException as e:
        status_code = ErrorHandler.get_http_status(e)
        raise HTTPException(status_code=status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in summarization endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/summarize_batch", response_model=SummarizationResponse)
async def summarize_batch(
    request: SummarizationBatchRequest,
) -> SummarizationResponse:
    logger.debug(f"Summarization batch request by model: {request.model}")
    try:
        summary: SummarizationResponse = await TextSummarizer.summarize_batch_async(
            request
        )
        return summary
    except FloudsBaseException as e:
        status_code = ErrorHandler.get_http_status(e)
        raise HTTPException(status_code=status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in batch summarization endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")
