from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """
    Request model for text summarization.
    """

    input: str = Field(..., description="The input text to be summarized.")
    model: str = Field(
        "sentence-t5-base",
        description="The model name to use for summarization. Defaults to 'sentence-t5-base'.",
    )
    projected_dimension: int = Field(
        256,
        description="The dimension to which the embedding will be projected. Defaults to 256.",
    )