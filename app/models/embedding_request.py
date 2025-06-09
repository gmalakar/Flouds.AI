from pydantic import Field

from app.models.base_request import BaseRequest


class EmbeddingBaseRequest(BaseRequest):
    projected_dimension: int = Field(
        128,
        description="The dimension to which the embedding will be projected. Defaults to 128.",
    )


class EmbeddingRequest(EmbeddingBaseRequest):
    """
    Request model for text embedding.
    """

    input: str = Field(..., description="The input text to be embedded.")


class EmbeddingBatchRequest(EmbeddingBaseRequest):
    """
    Request model for batch text embedding.
    """

    inputs: list[str] = Field(..., description="The input texts to be embedded.")
