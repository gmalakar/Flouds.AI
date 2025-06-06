from fastapi import APIRouter
from pydantic import BaseModel

from app.logger import get_logger
from app.models.summarization_request import SummarizationRequest
from app.services.summarizer_service import TextSummarizer


class SummarizationResponse(BaseModel):
    summary: str


router = APIRouter()
logger = get_logger("router")


@router.post("/summarize", tags=["summarize"], response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    logger.info(f"Summarization request by model: {request.model_to_use}")
    summary = TextSummarizer.summarize(request.model_to_use, request.text)
    return {"summary": summary}
