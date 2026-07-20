from pydantic import BaseModel
from typing import Dict, Any, Optional

class HealthCheckSchema(BaseModel):
    status: str
    model_loaded: bool
    version: str
    details: Optional[Dict[str, Any]] = None
