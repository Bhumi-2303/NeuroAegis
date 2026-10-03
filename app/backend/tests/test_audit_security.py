from __future__ import annotations
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.core.auth import create_access_token, get_password_hash
from app.core.config import settings
from app.db.database import Base, SessionLocal, engine, ensure_schema_compatibility
from app.db.models import AuditEvent, Patient, PredictionJob, Tenant, User
from app.main import app

TENANT_SEC_A = "tenant_audit_sec_a"
TENANT_SEC_B = "tenant_audit_sec_b"
USER_ADMIN_A = "user_sec_admin_a"
USER_CLIN_A = "user_sec_clin_a"
USER_RES_A = "user_sec_res_a"
USER_CLIN_B = "user_sec_clin_b"

PATIENT_SEC_B = "patient_audit_sec_b"
JOB_SEC_B = "job_audit_sec_b"


@pytest.fixture(scope="module", autouse=True)
def setup_security_audit_db():
    Base.metadata.create_all(bind=engine)
    ensure_schema_compatibility(engine)
    db = SessionLocal()
    try:
        # Tenants
        for t_id, t_name in [(TENANT_SEC_A, "Sec Clinic A"), (TENANT_SEC_B, "Sec Clinic B")]:
            if not db.query(Tenant).filter(Tenant.id == t_id).first():
                db.add(Tenant(id=t_id, name=t_name, slug=t_id, is_active=True))

        # Users
        users = [
            (USER_ADMIN_A, TENANT_SEC_A, "sec_admin_a", "admin"),
            (USER_CLIN_A, TENANT_SEC_A, "sec_clin_a", "clinician"),
            (USER_RES_A, TENANT_SEC_A, "sec_res_a", "researcher"),
            (USER_CLIN_B, TENANT_SEC_B, "sec_clin_b", "clinician"),
        ]
        for u_id, t_id, u_name, role in users:
            if not db.query(User).filter(User.id == u_id).first():
                db.add(
                    User(
                        id=u_id,
                        tenant_id=t_id,
                        username=u_name,
                        hashed_password=get_password_hash("ValidPass123!"),
                        role=role,
                        is_active=True,
                    )
                )

        # Patient in Tenant B
        if not db.query(Patient).filter(Patient.id == PATIENT_SEC_B).first():
            db.add(
                Patient(
                    id=PATIENT_SEC_B,
                    tenant_id=TENANT_SEC_B,
                    name="Patient B Sec",
                    age=45,
                    gender="M",
                    weight=70.0,
                    height=175.0,
                )
            )

        # Job in Tenant B
        if not db.query(PredictionJob).filter(PredictionJob.id == JOB_SEC_B).first():
            db.add(
                PredictionJob(
                    id=JOB_SEC_B,
                    tenant_id=TENANT_SEC_B,
                    patient_id=PATIENT_SEC_B,
                    status="completed",
                    prediction_label="Non-Seizure",
                    probability_seizure=0.1,
                )
            )

        db.commit()
    finally:
        db.close()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def make_token(user_id: str, role: str, tenant_id: str) -> str:
    db = SessionLocal()
    try:
        db_user = db.query(User).filter(User.id == user_id).first()
        tv = db_user.token_version if db_user else 1
    finally:
        db.close()
    user = User(
        id=user_id,
        username=f"u_{user_id}",
        tenant_id=tenant_id,
        role=role,
        token_version=tv,
    )
    return create_access_token(user)


def test_audit_login_success(client):
    """Verify auth.login.success is recorded with user identity, tenant, role, and no password."""
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "sec_clin_a", "password": "ValidPass123!"},
    )
    assert response.status_code == 200

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.login.success",
                AuditEvent.actor_id == USER_CLIN_A,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "success"
        assert event.actor_type == "user"
        assert event.tenant_id == TENANT_SEC_A
        meta = event.metadata_json or {}
        assert meta.get("username") == "sec_clin_a"
        assert meta.get("role") == "clinician"
        # Invariant: password never logged or stored
        assert "password" not in str(meta).lower()
        assert "validpass123" not in str(meta).lower()
    finally:
        db.close()


def test_audit_login_failure(client):
    """Verify auth.login.failure is anonymous, has no tenant/actor_id, records attempted_username, and no password."""
    attempted = f"failed_user_{uuid.uuid4().hex[:6]}"
    bad_pass = "WrongPassword_Secret999!"
    response = client.post(
        "/api/v1/auth/login",
        json={"username": attempted, "password": bad_pass},
    )
    assert response.status_code == 401

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.login.failure",
                AuditEvent.outcome == "failure",
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.actor_type == "anonymous"
        assert event.actor_id is None
        assert event.tenant_id is None
        meta = event.metadata_json or {}
        assert meta.get("attempted_username") == attempted
        # Invariant: password never logged or stored
        assert bad_pass not in str(meta)
        assert "password" not in str(meta).lower()
    finally:
        db.close()


def test_audit_logout(client):
    """Verify auth.logout records authenticated actor, tenant, and success outcome."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "sec_clin_a", "password": "ValidPass123!"},
    )
    assert login_resp.status_code == 200
    csrf_token = login_resp.json().get("csrf_token")

    # Logout with CSRF token
    headers = {"X-CSRF-Token": csrf_token}
    logout_resp = client.post("/api/v1/auth/logout", headers=headers)
    assert logout_resp.status_code == 200

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.logout",
                AuditEvent.actor_id == USER_CLIN_A,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "success"
        assert event.actor_type == "user"
        assert event.tenant_id == TENANT_SEC_A
    finally:
        db.close()


def test_audit_logout_all(client):
    """Verify auth.logout_all records revocation event."""
    token = make_token(USER_ADMIN_A, "admin", TENANT_SEC_A)
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post("/api/v1/auth/logout-all", headers=headers)
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.logout_all",
                AuditEvent.actor_id == USER_ADMIN_A,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "success"
        assert event.tenant_id == TENANT_SEC_A
    finally:
        db.close()


def test_audit_session_replay_detected(client):
    """Verify replay detection generates auth.session.replay_detected audit event."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "sec_clin_a", "password": "ValidPass123!"},
    )
    assert login_resp.status_code == 200
    csrf_token = login_resp.json().get("csrf_token")
    initial_refresh = login_resp.cookies.get(settings.REFRESH_COOKIE_NAME)
    assert initial_refresh is not None

    # Rotate refresh token once
    refresh_resp1 = client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf_token})
    assert refresh_resp1.status_code == 200
    new_csrf = refresh_resp1.json().get("csrf_token")

    # Replay the old refresh token (which has now been revoked and replaced)
    client.cookies.set(settings.REFRESH_COOKIE_NAME, initial_refresh, path=settings.REFRESH_COOKIE_PATH)
    replay_resp = client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": new_csrf})
    assert replay_resp.status_code == 401

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.session.replay_detected",
                AuditEvent.actor_id == USER_CLIN_A,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "denied"
        assert event.tenant_id == TENANT_SEC_A
    finally:
        db.close()


def test_audit_csrf_failure(client):
    """Verify missing/invalid CSRF token records auth.csrf.failure audit event."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "sec_clin_a", "password": "ValidPass123!"},
    )
    assert login_resp.status_code == 200

    # Post with cookie session but invalid CSRF header
    resp = client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": "invalid_csrf_val"},
    )
    assert resp.status_code == 403

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.csrf.failure",
                AuditEvent.outcome == "denied",
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "denied"
    finally:
        db.close()


def test_audit_rbac_access_denied(client):
    """Verify accessing restricted endpoint by non-admin records auth.access.denied."""
    # Clinician attempts to view admin-only audit logs
    token = make_token(USER_CLIN_A, "clinician", TENANT_SEC_A)
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.get("/api/v1/audit-logs", headers=headers)
    assert resp.status_code == 403

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.access.denied",
                AuditEvent.actor_id == USER_CLIN_A,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "denied"
        assert event.tenant_id == TENANT_SEC_A
        meta = event.metadata_json or {}
        assert meta.get("role") == "clinician"
    finally:
        db.close()


def test_audit_cross_tenant_violation_attempt(client):
    """
    Verify cross-tenant resource attempt:
    1. Returns uniform 404 (non-leaking error)
    2. Records auth.tenant_violation.attempt under CALLER's tenant_id and user
    3. Records target resource_id
    """
    token = make_token(USER_CLIN_A, "clinician", TENANT_SEC_A)
    headers = {"Authorization": f"Bearer {token}"}

    # Access patient belonging to Tenant B
    resp = client.get(f"/api/v1/patients/{PATIENT_SEC_B}", headers=headers)
    assert resp.status_code == 404

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.tenant_violation.attempt",
                AuditEvent.actor_id == USER_CLIN_A,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "denied"
        # MUST be recorded under caller's tenant
        assert event.tenant_id == TENANT_SEC_A
        assert event.resource_id == PATIENT_SEC_B
        assert event.resource_type == "patient"
    finally:
        db.close()


def test_audit_cross_tenant_data_deletion_actor_attributed(client):
    """Verify cross-tenant patient data deletion records actor_id=current_user.id (DEF-01)."""
    token = make_token(USER_ADMIN_A, "admin", TENANT_SEC_A)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.delete(f"/api/v1/data/patient/{PATIENT_SEC_B}", headers=headers)
    assert resp.status_code == 404

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.tenant_violation.attempt",
                AuditEvent.actor_id == USER_ADMIN_A,
                AuditEvent.resource_id == PATIENT_SEC_B,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "denied"
        assert event.actor_type == "user"
        assert event.actor_id == USER_ADMIN_A
        assert event.tenant_id == TENANT_SEC_A
        assert event.resource_type == "patient"
    finally:
        db.close()


def test_audit_cross_tenant_job_status_and_report_actor_attributed(client):
    """Verify cross-tenant v2 job status and report endpoints record actor_id=current_user.id (DEF-01)."""
    token = make_token(USER_CLIN_A, "clinician", TENANT_SEC_A)
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Job status endpoint
    status_resp = client.get(f"/api/v2/jobs/{JOB_SEC_B}", headers=headers)
    assert status_resp.status_code == 404

    db = SessionLocal()
    try:
        ev_status = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.tenant_violation.attempt",
                AuditEvent.actor_id == USER_CLIN_A,
                AuditEvent.resource_id == JOB_SEC_B,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert ev_status is not None
        assert ev_status.actor_type == "user"
        assert ev_status.actor_id == USER_CLIN_A
        assert ev_status.tenant_id == TENANT_SEC_A
        assert ev_status.resource_type == "prediction_job"
    finally:
        db.close()

    # 2. Job report endpoint
    report_resp = client.get(f"/api/v2/report/{JOB_SEC_B}", headers=headers)
    assert report_resp.status_code == 404

    db = SessionLocal()
    try:
        ev_report = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.tenant_violation.attempt",
                AuditEvent.actor_id == USER_CLIN_A,
                AuditEvent.resource_id == JOB_SEC_B,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert ev_report is not None
        assert ev_report.actor_type == "user"
        assert ev_report.actor_id == USER_CLIN_A
        assert ev_report.tenant_id == TENANT_SEC_A
    finally:
        db.close()


def test_audit_cross_tenant_jobs_v1_actor_attributed(client):
    """Verify cross-tenant v1 job endpoint records actor_id=current_user.id (DEF-01)."""
    token = make_token(USER_CLIN_A, "clinician", TENANT_SEC_A)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get(f"/api/v1/jobs/{JOB_SEC_B}", headers=headers)
    assert resp.status_code == 404

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.tenant_violation.attempt",
                AuditEvent.actor_id == USER_CLIN_A,
                AuditEvent.resource_id == JOB_SEC_B,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.actor_type == "user"
        assert event.actor_id == USER_CLIN_A
        assert event.tenant_id == TENANT_SEC_A
        assert event.resource_type == "prediction_job"
    finally:
        db.close()


def test_audit_cross_tenant_stream_actor_attributed(client):
    """Verify cross-tenant v1 stream endpoint records actor_id=current_user.id (DEF-01)."""
    token = make_token(USER_CLIN_A, "clinician", TENANT_SEC_A)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get(f"/api/v1/stream?job_id={JOB_SEC_B}", headers=headers)
    assert resp.status_code == 404

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "auth.tenant_violation.attempt",
                AuditEvent.actor_id == USER_CLIN_A,
                AuditEvent.resource_id == JOB_SEC_B,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.actor_type == "user"
        assert event.actor_id == USER_CLIN_A
        assert event.tenant_id == TENANT_SEC_A
        assert event.resource_type == "prediction_job"
    finally:
        db.close()
