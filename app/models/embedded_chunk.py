from pydantic import BaseModel, Field


class EmbededChunk(BaseModel):
    vector: list[float] = Field(
        ..., description="The generated embedding for the text chunk."
    )
    chunk: str = Field(..., description="The original text chunk that was embedded.")
