from uuid import UUID
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Job(BaseModel):
    """
    Domain entity representing an async processing job (e.g., pipeline execution).
    """
    id: UUID
    patient_id: UUID
    status: JobStatus = Field(default=JobStatus.QUEUED)
    progress: int = Field(default=0, ge=0, le=100)
    prediction_label: Optional[str] = None
    shap_explanation: Optional[Dict[str, Any]] = None
    pipeline_trace: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
