from fastapi import APIRouter
from pydantic import BaseModel

from app.logger import get_logger
from app.models.summarization_request import SummarizationRequest
from app.models.summarization_response import SummarizationResponse
from app.services.summarizer_service import TextSummarizer

router = APIRouter()
logger = get_logger("router")


@router.post("/summarize", tags=["summarize"], response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    logger.debug(f"Summarization request by model: {request.model}")
    summary = TextSummarizer.summarize(request.model, request.input)
    return summary
