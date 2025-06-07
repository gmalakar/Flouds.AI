from pydantic import BaseModel
from pydantic import Field
from app.models.base_response import BaseResponse

class SummaryResults(BaseModel):
    summary: str

class SummarizationResponse(BaseResponse):
    """
    Response model for text summarization.
    """
    results: SummaryResults = Field(
        ..., description="The generated summary and related metadata as an object."
    )
# This class extends BaseResponse to include a results field, which is a SummaryResults object containing the summarized text and related metadata.
# The Field decorator is used to provide additional metadata for the results field, such as a description.