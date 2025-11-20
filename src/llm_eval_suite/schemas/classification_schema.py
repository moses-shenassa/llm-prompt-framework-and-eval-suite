from pydantic import BaseModel, Field
from typing import Literal


class ClassificationOutput(BaseModel):
    """Expected structure for classification outputs."""

    label: str = Field(..., description="Predicted label from allowed label set or 'unknown'.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence in the prediction.")
