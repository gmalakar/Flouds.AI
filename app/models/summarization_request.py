from pydantic import BaseModel, Field


class SummarizationBaseRequest(BaseModel):
    model: str = Field(
        "t5-small",
        description="The model name to use for summarization. Defaults to 't5-small'.",
    )
    use_optimized_model: bool = Field(
        False,
        description="Whether to use an optimized ONNX model for summarization. Defaults to False.",
    )


class SummarizationRequest(SummarizationBaseRequest):
    input: str = Field(..., description="The input text to be summarized.")


class SummarizationBatchRequest(SummarizationBaseRequest):
    inputs: list[str] = Field(..., description="The input texts to be summarized.")
