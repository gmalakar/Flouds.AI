from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from app.logger import get_logger
from app.models.embedding_request import EmbeddingRequest
from app.models.embedding_response import EmbeddingResponse
from app.services.embedder_service import SentenceTransformer

router = APIRouter()
logger = get_logger("router")


@router.post("/embed", tags=["embedder"], response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    logger.debug(f"Embedding request by model: {request.model}")
    response = SentenceTransformer.embed_text(
        text=request.input,
        model_to_use=request.model,
        projected_dimension=request.projected_dimension,
    )
    return response
