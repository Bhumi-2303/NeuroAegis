from pydantic import BaseModel
from typing import Dict, Any, Optional

class HealthCheckSchema(BaseModel):
    status: str
    model_loaded: bool
    version: str
    model_version: Optional[str] = None
    dataset_name: Optional[str] = None
    last_load_time: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
