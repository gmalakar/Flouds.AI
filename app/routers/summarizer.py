from typing import List

from fastapi import APIRouter

from app.logger import get_logger
from app.models.summarization_request import (
    SummarizationBatchRequest,
    SummarizationRequest,
)
from app.models.summarization_response import SummarizationResponse
from app.services.summarizer_service import TextSummarizer

router = APIRouter()
logger = get_logger("router")

# HINTS:
# - Both endpoints are async and call async methods from TextSummarizer.
# - The /summarize endpoint expects a SummarizationRequest and returns a SummarizationResponse.
# - The /summarize_batch endpoint expects a SummarizationBatchRequest and returns a list of SummarizationResponse.
# - Use type hints for FastAPI endpoint parameters and return types for better validation and editor support.


@router.post("/summarize", tags=["summarize"], response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest) -> SummarizationResponse:
    logger.debug(f"Summarization request by model: {request.model}")
    summary: SummarizationResponse = TextSummarizer.summarize(request)
    return summary


@router.post(
    "/summarize_batch", tags=["summarize"], response_model=List[SummarizationResponse]
)
async def summarize_batch(
    request: SummarizationBatchRequest,
) -> List[SummarizationResponse]:
    logger.debug(f"Summarization batch request by model: {request.model}")
    summary: List[SummarizationResponse] = await TextSummarizer.summarize_batch_async(
        request
    )
    return summary
