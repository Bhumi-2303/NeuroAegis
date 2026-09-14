from __future__ import annotations
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PatientBase(BaseModel):
    name: str
    age: int
    gender: str
    weight: float
    height: float
    medical_history: str | None = None
    vital_signs: dict[str, Any] | None = None
    status: str = "active"
    last_visit: datetime | None = None

class PatientCreate(PatientBase):
    pass

class PatientResponse(PatientBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True
