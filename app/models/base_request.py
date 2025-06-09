from pydantic import BaseModel, Field, condecimal


class BaseRequest(BaseModel):
    """
    Request model for text summarization.
    """

    model: str = Field(
        ...,
        description="The model name to use for summarization. This field is required.",
    )