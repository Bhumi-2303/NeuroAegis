from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


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
    prediction_label: str | None = None
    shap_explanation: dict[str, Any] | None = None
    pipeline_trace: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
