from pydantic import BaseModel, Field
from typing import List, Literal


class SummarizationOutput(BaseModel):
    """Expected structure for summarization outputs."""

    summary: str = Field(..., description="Short, plain-language summary of the input text.")
    reading_level: Literal["children", "teen", "adult"] = Field(
        ..., description="Intended reading level for the summary."
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="List of key points captured from the source text.",
    )
