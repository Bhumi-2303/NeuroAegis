from __future__ import annotations
import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.audit import (
    SAFE_AUDIT_EVENT_TYPES,
    SAFE_AUDIT_OUTCOMES,
    validate_identifier,
)
from app.core.auth import require_roles
from app.db.database import get_db
from app.db.models import AuditEvent, User
from app.schemas.audit import AuditEventResponse

router = APIRouter()

SAFE_PARAM_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def _validate_filter_id(val: str | None, name: str) -> str | None:
    if val is None:
        return None
    val_clean = val.strip()
    if not val_clean or not SAFE_PARAM_ID_REGEX.match(val_clean):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {name} format",
        )
    return val_clean


@router.get("", response_model=list[AuditEventResponse])
@router.get("/", response_model=list[AuditEventResponse])
def get_audit_logs(
    event_type: Optional[str] = Query(None, description="Filter by exact event type"),
    outcome: Optional[str] = Query(None, description="Filter by event outcome: success, failure, denied, error"),
    start_time: Optional[datetime] = Query(None, description="Filter events on or after UTC datetime"),
    end_time: Optional[datetime] = Query(None, description="Filter events on or before UTC datetime"),
    patient_id: Optional[str] = Query(None, description="Filter by historical patient ID"),
    job_id: Optional[str] = Query(None, description="Filter by historical prediction job ID"),
    limit: int = Query(50, ge=1, le=100, description="Page size (1-100, default 50)"),
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("admin")),
):
    """
    Retrieve tenant-scoped audit records.
    Accessible strictly to tenant administrators.
    Mandatory filter: AuditEvent.tenant_id == current_user.tenant_id.
    """
    # 1. Base query strictly scoped to caller's authenticated tenant
    query = db.query(AuditEvent).filter(AuditEvent.tenant_id == current_user.tenant_id)

    # 2. Bounded event_type filter
    if event_type is not None:
        clean_event = event_type.strip()
        if clean_event not in SAFE_AUDIT_EVENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown or invalid event_type filter: '{clean_event}'",
            )
        query = query.filter(AuditEvent.event_type == clean_event)

    # 3. Bounded outcome filter
    if outcome is not None:
        clean_outcome = outcome.strip().lower()
        if clean_outcome not in SAFE_AUDIT_OUTCOMES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown or invalid outcome filter: '{clean_outcome}'",
            )
        query = query.filter(AuditEvent.outcome == clean_outcome)

    # 4. Date range filters
    if start_time is not None:
        query = query.filter(AuditEvent.occurred_at >= start_time)
    if end_time is not None:
        query = query.filter(AuditEvent.occurred_at <= end_time)

    # 5. Identifier filters
    if patient_id is not None:
        clean_patient_id = _validate_filter_id(patient_id, "patient_id")
        query = query.filter(AuditEvent.patient_id == clean_patient_id)

    if job_id is not None:
        clean_job_id = _validate_filter_id(job_id, "job_id")
        query = query.filter(AuditEvent.job_id == clean_job_id)

    # 6. Reverse-chronological order and pagination
    records = query.order_by(AuditEvent.occurred_at.desc()).offset(skip).limit(limit).all()

    # Map ORM objects to response schemas
    results: list[AuditEventResponse] = []
    for rec in records:
        results.append(
            AuditEventResponse(
                id=rec.id,
                occurred_at=rec.occurred_at,
                event_type=rec.event_type,
                outcome=rec.outcome,
                actor_type=rec.actor_type,
                actor_id=rec.actor_id,
                tenant_id=rec.tenant_id,
                request_id=rec.request_id,
                session_id=rec.session_id,
                resource_type=rec.resource_type,
                resource_id=rec.resource_id,
                patient_id=rec.patient_id,
                job_id=rec.job_id,
                error_category=rec.error_category,
                metadata=rec.metadata_json,
            )
        )
    return results
