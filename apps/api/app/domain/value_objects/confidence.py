from enum import Enum
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing_extensions import Self

class ConfidenceBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ConfidenceScore(BaseModel):
    """
    Value object representing the confidence of a prediction.
    """
    value: float = Field(..., description="Confidence probability between 0.0 and 1.0")
    band: ConfidenceBand = Field(..., description="Categorical band of the confidence")

    @model_validator(mode="after")
    def validate_value_and_band(self) -> Self:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Confidence value must be between 0.0 and 1.0")
            
        # Optional: ensure band logic is consistent, e.g.:
        # if self.value < 0.5 and self.band != ConfidenceBand.LOW:
        #    raise ValueError("Mismatch between value and band")
        return self
    
    model_config = ConfigDict(frozen=True)
