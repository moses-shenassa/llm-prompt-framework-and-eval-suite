from pydantic import BaseModel, Field
from typing import List, Optional


class ExtractionOutput(BaseModel):
    """Expected structure for extraction outputs."""

    name: Optional[str] = Field(None, description="Person's name if present.")
    age: Optional[int] = Field(None, description="Age if explicitly mentioned.")
    location: Optional[str] = Field(None, description="Location if mentioned.")
    conditions: List[str] = Field(default_factory=list, description="Medical or relevant conditions.")
    medications: List[str] = Field(default_factory=list, description="Medications or treatments.")
