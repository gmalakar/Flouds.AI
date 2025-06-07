from typing import List

from pydantic import Field

from app.models.base_response import BaseResponse
from app.models.embedded_chunk import EmbededChunk


class EmbeddingResponse(BaseResponse):
    """
    Response model for text embedding.
    """

    results: List[EmbededChunk] = Field(
        ..., description="A list of embedding chunks for the input text."
    )
