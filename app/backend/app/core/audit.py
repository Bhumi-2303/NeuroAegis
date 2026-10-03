from __future__ import annotations
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.errors import (
    SAFE_ERROR_GENERIC,
    SAFE_JOB_ERRORS,
    sanitize_error_text,
    sanitize_for_log,
)
from app.core.logging import get_request_id, validate_or_generate_request_id

logger = logging.getLogger("neuroaegis.audit")

# Bounded Audit Event Taxonomy (Phase 10.4)
EVENT_AUTH_LOGIN_SUCCESS = "auth.login.success"
EVENT_AUTH_LOGIN_FAILURE = "auth.login.failure"
EVENT_AUTH_LOGOUT = "auth.logout"
EVENT_AUTH_LOGOUT_ALL = "auth.logout_all"
EVENT_AUTH_SESSION_ROTATED = "auth.session.rotated"
EVENT_AUTH_SESSION_REPLAY_DETECTED = "auth.session.replay_detected"
EVENT_AUTH_CSRF_FAILURE = "auth.csrf.failure"
EVENT_AUTH_ACCESS_DENIED = "auth.access.denied"
EVENT_AUTH_TENANT_VIOLATION = "auth.tenant_violation.attempt"

EVENT_CLINICAL_PATIENT_CREATED = "clinical.patient.created"
EVENT_CLINICAL_PATIENT_VIEWED = "clinical.patient.viewed"
EVENT_CLINICAL_PATIENT_DELETED = "clinical.patient.deleted"
EVENT_CLINICAL_JOB_CREATED = "clinical.job.created"
EVENT_CLINICAL_JOB_REPORT_VIEWED = "clinical.job.report_viewed"
EVENT_CLINICAL_PREDICTION_COMPLETED = "clinical.prediction.completed"
EVENT_CLINICAL_PREDICTION_FAILED = "clinical.prediction.failed"

EVENT_SYSTEM_WORKER_LEASE_EXPIRED = "system.worker.lease_expired"

SAFE_AUDIT_EVENT_TYPES: set[str] = {
    EVENT_AUTH_LOGIN_SUCCESS,
    EVENT_AUTH_LOGIN_FAILURE,
    EVENT_AUTH_LOGOUT,
    EVENT_AUTH_LOGOUT_ALL,
    EVENT_AUTH_SESSION_ROTATED,
    EVENT_AUTH_SESSION_REPLAY_DETECTED,
    EVENT_AUTH_CSRF_FAILURE,
    EVENT_AUTH_ACCESS_DENIED,
    EVENT_AUTH_TENANT_VIOLATION,
    EVENT_CLINICAL_PATIENT_CREATED,
    EVENT_CLINICAL_PATIENT_VIEWED,
    EVENT_CLINICAL_PATIENT_DELETED,
    EVENT_CLINICAL_JOB_CREATED,
    EVENT_CLINICAL_JOB_REPORT_VIEWED,
    EVENT_CLINICAL_PREDICTION_COMPLETED,
    EVENT_CLINICAL_PREDICTION_FAILED,
    EVENT_SYSTEM_WORKER_LEASE_EXPIRED,
}

SAFE_AUDIT_OUTCOMES: set[str] = {"success", "failure", "denied", "error"}
SAFE_AUDIT_ACTOR_TYPES: set[str] = {"user", "worker", "system", "anonymous"}
SAFE_AUDIT_RESOURCE_TYPES: set[str] = {"patient", "prediction_job", "session", "user", "system"}

_SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-.:]{1,128}$")
_CONTROL_CHAR_REGEX = re.compile(r"[\r\n\t\x00-\x1f\x7f]")

# Strict metadata allowlist
ALLOWED_METADATA_KEYS: set[str] = {
    "username",
    "attempted_username",
    "role",
    "dataset",
    "execution_mode",
    "error_detail",
    "client_ip_hash",
    "count",
    "batch_size",
    "model_name",
}

FORBIDDEN_METADATA_SUBSTRINGS: tuple[str, ...] = (
    "password",
    "passwd",
    "token",
    "secret",
    "jwt",
    "bearer",
    "cookie",
    "session_secret",
    "key",
    "eeg",
    "sample",
    "array",
    "history",
    "diagnosis",
    "traceback",
    "stack",
    "file://",
    "\\",
    ".py",
)


def validate_identifier(val: str | None, max_len: int = 64) -> str | None:
    """Validate that an ID string is bounded and alphanumeric with hyphens, underscores, dots, or colons."""
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str or len(val_str) > max_len or not _SAFE_ID_REGEX.match(val_str):
        raise ValueError(f"Invalid identifier format or length: '{sanitize_for_log(val_str[:32])}'")
    return val_str


def sanitize_bounded_text(text: str | None, max_len: int = 128) -> str | None:
    """Sanitize and bound a text field, stripping control characters and newlines."""
    if text is None:
        return None
    cleaned = _CONTROL_CHAR_REGEX.sub("", str(text)).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


def validate_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Validate and sanitize metadata against strict bounded rules:
    - Only allowed keys
    - No nested dicts or arbitrary structures
    - Primitive types only (str, int, float, bool)
    - Strings bounded to 128 chars
    - Max 10 keys
    - Total serialization <= 1024 chars
    - Forbidden substrings rejected
    """
    if not metadata:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")
    if len(metadata) > 10:
        raise ValueError("Metadata cannot contain more than 10 entries")

    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        key_str = str(key).strip().lower()
        if key_str not in ALLOWED_METADATA_KEYS:
            raise ValueError(f"Metadata key '{key_str}' is not permitted")

        if any(bad in key_str for bad in ("password", "token", "secret", "cookie")):
            raise ValueError(f"Metadata key '{key_str}' contains forbidden pattern")

        if value is None:
            sanitized[key_str] = None
        elif isinstance(value, bool):
            sanitized[key_str] = value
        elif isinstance(value, (int, float)):
            sanitized[key_str] = value
        elif isinstance(value, str):
            # Check forbidden substrings
            val_lower = value.lower()
            if any(bad in val_lower for bad in FORBIDDEN_METADATA_SUBSTRINGS):
                raise ValueError(f"Metadata value for '{key_str}' contains forbidden pattern")
            cleaned_val = sanitize_bounded_text(value, max_len=128)
            sanitized[key_str] = cleaned_val
        else:
            raise ValueError(f"Unsupported metadata value type for key '{key_str}': {type(value).__name__}")

    # Check serialized size
    serialized = json.dumps(sanitized)
    if len(serialized) > 1024:
        raise ValueError("Serialized metadata exceeds maximum size of 1024 characters")

    return sanitized


def validate_error_category(error_cat: str | None) -> str | None:
    """Ensure error category belongs to safe error taxonomy or sanitize it."""
    if error_cat is None:
        return None
    val = str(error_cat).strip()
    if val in SAFE_JOB_ERRORS:
        return val
    return sanitize_error_text(val, default=SAFE_ERROR_GENERIC)


def build_audit_event_kwargs(
    event_type: str,
    outcome: str,
    actor_type: str,
    actor_id: str | None = None,
    tenant_id: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    patient_id: str | None = None,
    job_id: str | None = None,
    error_category: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Validate all fields according to Phase 10.4 audit constraints
    and return validated kwargs ready for AuditEvent model instantiation.
    """
    if event_type not in SAFE_AUDIT_EVENT_TYPES:
        raise ValueError(f"Invalid audit event_type: '{event_type}'")

    if outcome not in SAFE_AUDIT_OUTCOMES:
        raise ValueError(f"Invalid audit outcome: '{outcome}'")

    if actor_type not in SAFE_AUDIT_ACTOR_TYPES:
        raise ValueError(f"Invalid audit actor_type: '{actor_type}'")

    if resource_type is not None and resource_type not in SAFE_AUDIT_RESOURCE_TYPES:
        raise ValueError(f"Invalid audit resource_type: '{resource_type}'")

    # Invariant: anonymous actor MUST NOT have actor_id or tenant_id
    if actor_type == "anonymous":
        actor_id = None
        tenant_id = None

    valid_actor_id = validate_identifier(actor_id, max_len=128)
    valid_tenant_id = validate_identifier(tenant_id)

    # Use ContextVar request_id if not explicitly provided
    effective_req_id = request_id or get_request_id()
    if effective_req_id:
        effective_req_id = validate_or_generate_request_id(effective_req_id)

    valid_session_id = validate_identifier(session_id)
    valid_resource_id = validate_identifier(resource_id, max_len=128)
    valid_patient_id = validate_identifier(patient_id)
    valid_job_id = validate_identifier(job_id)

    valid_error_category = validate_error_category(error_category)
    valid_metadata = validate_metadata(metadata)

    event_id = str(uuid.uuid4())
    now_utc = datetime.now(timezone.utc)

    return {
        "id": event_id,
        "occurred_at": now_utc,
        "event_type": event_type,
        "outcome": outcome,
        "actor_type": actor_type,
        "actor_id": valid_actor_id,
        "tenant_id": valid_tenant_id,
        "request_id": effective_req_id,
        "session_id": valid_session_id,
        "resource_type": resource_type,
        "resource_id": valid_resource_id,
        "patient_id": valid_patient_id,
        "job_id": valid_job_id,
        "error_category": valid_error_category,
        "metadata_json": valid_metadata,
    }


def record_audit_event(
    db: Session,
    event_type: str,
    outcome: str,
    actor_type: str,
    actor_id: str | None = None,
    tenant_id: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    patient_id: str | None = None,
    job_id: str | None = None,
    error_category: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """
    Constructs an AuditEvent and adds it to the provided DB session.
    Used for business mutations where the audit record is committed
    atomically with the entity mutation (Case 1: Business Mutation).
    Does NOT commit the session; caller commits.
    """
    from app.db.models import AuditEvent

    kwargs = build_audit_event_kwargs(
        event_type=event_type,
        outcome=outcome,
        actor_type=actor_type,
        actor_id=actor_id,
        tenant_id=tenant_id,
        request_id=request_id,
        session_id=session_id,
        resource_type=resource_type,
        resource_id=resource_id,
        patient_id=patient_id,
        job_id=job_id,
        error_category=error_category,
        metadata=metadata,
    )
    event_record = AuditEvent(**kwargs)
    db.add(event_record)
    return event_record


def record_security_event(
    event_type: str,
    outcome: str,
    actor_type: str,
    actor_id: str | None = None,
    tenant_id: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    patient_id: str | None = None,
    job_id: str | None = None,
    error_category: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Records a security event in an isolated, dedicated short-lived DB transaction.
    Used for Case 2: Security Rejections (failed login, CSRF failure, 403, cross-tenant attempt).
    Commits immediately.
    If database persistence fails, logs an emergency error via sanitized logger
    and does NOT raise, ensuring the security rejection still returns properly.
    """
    from app.db.database import SessionLocal
    from app.db.models import AuditEvent

    db = SessionLocal()
    try:
        kwargs = build_audit_event_kwargs(
            event_type=event_type,
            outcome=outcome,
            actor_type=actor_type,
            actor_id=actor_id,
            tenant_id=tenant_id,
            request_id=request_id,
            session_id=session_id,
            resource_type=resource_type,
            resource_id=resource_id,
            patient_id=patient_id,
            job_id=job_id,
            error_category=error_category,
            metadata=metadata,
        )
        event_record = AuditEvent(**kwargs)
        db.add(event_record)
        db.commit()
        return event_record.id
    except Exception as exc:
        db.rollback()
        logger.error(
            f"Emergency: Failed to persist security audit event '{event_type}': {sanitize_for_log(str(exc))}"
        )
        return None
    finally:
        db.close()
