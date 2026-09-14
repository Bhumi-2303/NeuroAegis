from __future__ import annotations
import math
from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class Prediction(BaseModel):
    """
    Domain entity representing an ML prediction output.
    """
    id: UUID
    recording_id: UUID
    patient_id: UUID
    model_version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    label: str
    probabilities: dict[str, float]

    @model_validator(mode="after")
    def validate_probabilities(self) -> Self:
        total = sum(self.probabilities.values())
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")
        return self
