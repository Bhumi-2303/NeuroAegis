from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class PatientBase(BaseModel):
    name: str
    age: int
    gender: str
    weight: float
    height: float
    medical_history: Optional[Dict[str, Any]] = None
    vital_signs: Optional[Dict[str, Any]] = None

class PatientCreate(PatientBase):
    pass

class PatientResponse(PatientBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True
