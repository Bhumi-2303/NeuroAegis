from __future__ import annotations
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuditEventResponse(BaseModel):
    """
    Safe public response schema for audit log entries.
    Strictly excludes passwords, tokens, credentials, EEG data, and raw PHI.
    """
    id: str = Field(..., description="Unique event identifier (UUID)")
    occurred_at: datetime = Field(..., description="UTC timestamp of the audit event")
    event_type: str = Field(..., description="Machine-readable bounded event type")
    outcome: str = Field(..., description="Outcome of the event: success, failure, denied, error")
    actor_type: str = Field(..., description="Principal category: user, worker, system, anonymous")
    actor_id: str | None = Field(None, description="Identifier of the actor or None")
    tenant_id: str | None = Field(None, description="Authoritative tenant identifier or None")
    request_id: str | None = Field(None, description="Correlated request identifier or None")
    session_id: str | None = Field(None, description="Correlated session identifier or None")
    resource_type: str | None = Field(None, description="Target resource type or None")
    resource_id: str | None = Field(None, description="Target resource identifier or None")
    patient_id: str | None = Field(None, description="Historical patient reference or None")
    job_id: str | None = Field(None, description="Historical prediction job reference or None")
    error_category: str | None = Field(None, description="Safe error category or None")
    metadata: dict[str, Any] | None = Field(None, description="Sanitized, allowlisted contextual metadata")

    model_config = ConfigDict(from_attributes=True)
