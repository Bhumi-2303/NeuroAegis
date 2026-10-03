from __future__ import annotations
import uuid
import pytest
from fastapi.testclient import TestClient

from app.core.audit import (
    ALLOWED_METADATA_KEYS,
    FORBIDDEN_METADATA_SUBSTRINGS,
    record_audit_event,
    validate_identifier,
    validate_metadata,
)
from app.core.auth import create_access_token, get_password_hash
from app.db.database import Base, SessionLocal, engine, ensure_schema_compatibility
from app.db.models import AuditEvent, Tenant, User
from app.main import app

CLIENT_ADV = TestClient(app)

TENANT_ADV_A = "tenant_adv_alpha"
TENANT_ADV_B = "tenant_adv_beta"
USER_ADMIN_ADV_A = "user_adv_admin_a"
USER_ADMIN_ADV_B = "user_adv_admin_b"
USER_CLIN_ADV_A = "user_adv_clin_a"
USER_RES_ADV_A = "user_adv_res_a"


@pytest.fixture(scope="module", autouse=True)
def setup_adversarial_audit_db():
    Base.metadata.create_all(bind=engine)
    ensure_schema_compatibility(engine)
    db = SessionLocal()
    try:
        for t_id, name in [(TENANT_ADV_A, "Adv Clinic A"), (TENANT_ADV_B, "Adv Clinic B")]:
            if not db.query(Tenant).filter(Tenant.id == t_id).first():
                db.add(Tenant(id=t_id, name=name, slug=t_id, is_active=True))

        users = [
            (USER_ADMIN_ADV_A, TENANT_ADV_A, "adv_admin_a", "admin"),
            (USER_ADMIN_ADV_B, TENANT_ADV_B, "adv_admin_b", "admin"),
            (USER_CLIN_ADV_A, TENANT_ADV_A, "adv_clin_a", "clinician"),
            (USER_RES_ADV_A, TENANT_ADV_A, "adv_res_a", "researcher"),
        ]
        for u_id, t_id, u_name, role in users:
            if not db.query(User).filter(User.id == u_id).first():
                db.add(
                    User(
                        id=u_id,
                        tenant_id=t_id,
                        username=u_name,
                        hashed_password=get_password_hash("AdvPass123!"),
                        role=role,
                        is_active=True,
                    )
                )

        # Seed events in both tenants
        record_audit_event(
            db=db,
            event_type="auth.login.success",
            outcome="success",
            actor_type="user",
            actor_id=USER_ADMIN_ADV_A,
            tenant_id=TENANT_ADV_A,
        )
        record_audit_event(
            db=db,
            event_type="auth.login.success",
            outcome="success",
            actor_type="user",
            actor_id=USER_ADMIN_ADV_B,
            tenant_id=TENANT_ADV_B,
        )
        db.commit()
    finally:
        db.close()


def make_token(user_id: str, role: str, tenant_id: str) -> str:
    user = User(
        id=user_id,
        username=f"u_{user_id}",
        tenant_id=tenant_id,
        role=role,
        token_version=1,
    )
    return create_access_token(user)


def test_audit_spoofed_tenant_header_ignored():
    """Verify that spoofed X-Tenant-ID header or body tenant_id is ignored; audit uses authenticated principal."""
    token = make_token(USER_CLIN_ADV_A, "clinician", TENANT_ADV_A)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Tenant-ID": TENANT_ADV_B,  # Attacker tries to attribute to Tenant B
    }

    payload = {
        "name": "Spoof Test Patient",
        "age": 33,
        "gender": "M",
        "weight": 70.0,
        "height": 175.0,
        "tenant_id": TENANT_ADV_B,  # Body spoofing attempt
    }
    resp = CLIENT_ADV.post("/api/v1/patients/", json=payload, headers=headers)
    assert resp.status_code == 201
    created_id = resp.json()["id"]

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "clinical.patient.created",
                AuditEvent.patient_id == created_id,
            )
            .first()
        )
        assert event is not None
        # MUST belong to authenticated caller's tenant A, NEVER spoofed tenant B
        assert event.tenant_id == TENANT_ADV_A
        assert event.actor_id == USER_CLIN_ADV_A
    finally:
        db.close()


def test_audit_cross_tenant_query_forbidden():
    """Verify Admin in Tenant A cannot retrieve audit events for Tenant B, even with explicit param."""
    token_a = make_token(USER_ADMIN_ADV_A, "admin", TENANT_ADV_A)
    headers = {"Authorization": f"Bearer {token_a}"}

    # Request audit logs
    resp = CLIENT_ADV.get("/api/v1/audit-logs", headers=headers)
    assert resp.status_code == 200
    events = resp.json()
    for ev in events:
        assert ev["tenant_id"] == TENANT_ADV_A

    # Try requesting Tenant B logs specifically
    resp_b = CLIENT_ADV.get(f"/api/v1/audit-logs?tenant_id={TENANT_ADV_B}", headers=headers)
    assert resp_b.status_code == 200
    events_b = resp_b.json()
    # The endpoint must ignore or override the query param with caller's tenant
    for ev in events_b:
        assert ev["tenant_id"] == TENANT_ADV_A


def test_audit_metadata_injection_sanitization():
    """Verify that metadata validation strictly rejects unallowed keys, nested objects, and forbidden strings."""
    # 1. Non-whitelisted key
    with pytest.raises(ValueError, match="not permitted"):
        validate_metadata({"unknown_custom_key": "some_value"})

    # 2. Forbidden substrings in key or value
    with pytest.raises(ValueError, match="forbidden pattern"):
        validate_metadata({"username": "user; drop table users; -- password"})

    with pytest.raises(ValueError, match="forbidden pattern"):
        validate_metadata({"username": "bearer eyJhbGciOi..."})

    # 3. Nested dictionary
    with pytest.raises(ValueError, match="Unsupported metadata value type"):
        validate_metadata({"username": {"nested": "dict"}})

    # 4. Oversized string value (> 128 chars) gets truncated safely
    long_username = "a" * 150
    sanitized = validate_metadata({"username": long_username})
    assert len(sanitized["username"]) == 128

    # 5. Over 10 keys
    with pytest.raises(ValueError, match="cannot contain more than 10 entries"):
        validate_metadata({f"key_{i}": i for i in range(11)})


def test_audit_pagination_limits_enforced():
    """Verify that audit logs endpoint enforces max 100 limit (rejecting >100 with 422)."""
    token = make_token(USER_ADMIN_ADV_A, "admin", TENANT_ADV_A)
    headers = {"Authorization": f"Bearer {token}"}

    # Excess limit is rejected by validation schema
    resp = CLIENT_ADV.get("/api/v1/audit-logs?limit=500", headers=headers)
    assert resp.status_code == 422

    # Valid bounded limit works as expected
    resp_valid = CLIENT_ADV.get("/api/v1/audit-logs?limit=100", headers=headers)
    assert resp_valid.status_code == 200
    assert len(resp_valid.json()) <= 100


def test_audit_unauthorized_user_forbidden():
    """Verify that non-admin users and unauthenticated users cannot access /api/v1/audit-logs."""
    # Unauthenticated
    resp_unauth = CLIENT_ADV.get("/api/v1/audit-logs")
    assert resp_unauth.status_code == 401

    # Clinician
    token_clin = make_token(USER_CLIN_ADV_A, "clinician", TENANT_ADV_A)
    resp_clin = CLIENT_ADV.get("/api/v1/audit-logs", headers={"Authorization": f"Bearer {token_clin}"})
    assert resp_clin.status_code == 403

    # Researcher
    token_res = make_token(USER_RES_ADV_A, "researcher", TENANT_ADV_A)
    resp_res = CLIENT_ADV.get("/api/v1/audit-logs", headers={"Authorization": f"Bearer {token_res}"})
    assert resp_res.status_code == 403


def test_audit_orm_metadata_validation_direct():
    """
    Verify that direct ORM instantiation of AuditEvent enforces metadata validation (DEF-02).
    Bypassing the centralized record_audit_event() helper must still trigger validation.
    """
    # 1. Unknown metadata key rejected by ORM constructor
    with pytest.raises(ValueError, match="not permitted"):
        AuditEvent(
            id="orm-val-1",
            event_type="auth.login.success",
            outcome="success",
            actor_type="user",
            metadata={"unknown_custom_key": "some_value"},
        )

    # 2. Forbidden metadata pattern rejected by ORM constructor
    with pytest.raises(ValueError, match="forbidden pattern"):
        AuditEvent(
            id="orm-val-2",
            event_type="auth.login.success",
            outcome="success",
            actor_type="user",
            metadata={"username": "secret_password_123"},
        )

    # 3. Nested dictionary rejected by ORM constructor
    with pytest.raises(ValueError, match="Unsupported metadata value type"):
        AuditEvent(
            id="orm-val-3",
            event_type="auth.login.success",
            outcome="success",
            actor_type="user",
            metadata={"username": {"nested": "dict"}},
        )

    # 4. Attribute assignment also validates
    ev = AuditEvent(
        id="orm-val-4",
        event_type="auth.login.success",
        outcome="success",
        actor_type="user",
        metadata={"username": "valid_user"},
    )
    assert ev.metadata_json == {"username": "valid_user"}

    with pytest.raises(ValueError, match="not permitted"):
        ev.metadata_json = {"invalid_key": "fail"}

    # 5. Nullable metadata accepted
    ev_none = AuditEvent(
        id="orm-val-5",
        event_type="auth.login.success",
        outcome="success",
        actor_type="user",
        metadata=None,
    )
    assert ev_none.metadata_json is None
