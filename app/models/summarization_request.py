from pydantic import BaseModel, Field


class SummarizationRequest(BaseModel):
    """
    Request model for text summarization.
    """

    input: str = Field(..., description="The input text to be summarized.")
    model: str = Field(
        "t5-small",
        description="The model name to use for summarization. Defaults to 't5-small'.",
    )
