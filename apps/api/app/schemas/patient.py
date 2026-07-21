from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class PatientBase(BaseModel):
    name: str
    age: int
    gender: str
    weight: float
    height: float
    medical_history: Optional[str] = None
    vital_signs: Optional[Dict[str, Any]] = None
    status: str = "active"
    last_visit: Optional[datetime] = None

class PatientCreate(PatientBase):
    pass

class PatientResponse(PatientBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True
