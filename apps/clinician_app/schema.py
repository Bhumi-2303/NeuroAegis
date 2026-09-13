from pydantic import BaseModel
from typing import Dict

class ClinicianAnnotation(BaseModel):
    """
    Structured record per event matching a schema compatible with Track D's agreement metrics.
    Importance scales are mapped as: None (0), Low (1), Medium (2), High (3).
    """
    event_id: str
    clinician_id: str
    channel_importance: Dict[str, int]
    time_importance: Dict[str, int]
    frequency_importance: Dict[str, int]
