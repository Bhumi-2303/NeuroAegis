from __future__ import annotations
import uuid
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.core.audit import record_audit_event, record_security_event
from app.db.database import Base, SessionLocal, engine, ensure_schema_compatibility
from app.db.models import AuditEvent, Patient, Tenant


@pytest.fixture(scope="module", autouse=True)
def setup_integrity_audit_db():
    Base.metadata.create_all(bind=engine)
    ensure_schema_compatibility(engine)
    db = SessionLocal()
    try:
        t_id = "tenant_audit_integ"
        if not db.query(Tenant).filter(Tenant.id == t_id).first():
            db.add(Tenant(id=t_id, name="Integrity Tenant", slug=t_id, is_active=True))
            db.commit()
    finally:
        db.close()


def test_audit_append_only_update_rejected():
    """Verify that updating an audit event fails at both ORM level and database trigger level."""
    db = SessionLocal()
    try:
        # Create a valid audit event
        ev = record_audit_event(
            db=db,
            event_type="auth.login.success",
            outcome="success",
            actor_type="user",
            actor_id="user_integ_01",
            tenant_id="tenant_audit_integ",
        )
        db.commit()
        ev_id = ev.id

        # 1. ORM level rejection
        ev.outcome = "failure"
        with pytest.raises(Exception, match="forbidden"):
            db.commit()
        db.rollback()

        # 2. Raw SQL / Database Trigger level rejection
        with pytest.raises(DBAPIError):
            db.execute(
                text("UPDATE audit_events SET outcome = 'failure' WHERE id = :id"),
                {"id": ev_id},
            )
            db.commit()
        db.rollback()
    finally:
        db.close()


def test_audit_append_only_delete_rejected():
    """Verify that deleting an audit event fails at both ORM level and database trigger level."""
    db = SessionLocal()
    try:
        ev = record_audit_event(
            db=db,
            event_type="auth.login.success",
            outcome="success",
            actor_type="user",
            actor_id="user_integ_02",
            tenant_id="tenant_audit_integ",
        )
        db.commit()
        ev_id = ev.id

        # 1. ORM level rejection
        db.delete(ev)
        with pytest.raises(Exception, match="forbidden"):
            db.commit()
        db.rollback()

        # 2. Raw SQL / Database Trigger level rejection
        with pytest.raises(DBAPIError):
            db.execute(
                text("DELETE FROM audit_events WHERE id = :id"),
                {"id": ev_id},
            )
            db.commit()
        db.rollback()
    finally:
        db.close()


def test_audit_transaction_atomicity_rollback():
    """Verify that business audit events share the transaction and roll back if business transaction fails."""
    db = SessionLocal()
    test_id = f"pat_rollback_{uuid.uuid4().hex[:6]}"
    try:
        # Simulate business transaction that adds patient and records audit, then encounters failure
        p = Patient(
            id=test_id,
            tenant_id="tenant_audit_integ",
            name="Rollback Patient",
            age=50,
            gender="M",
        )
        db.add(p)
        ev = record_audit_event(
            db=db,
            event_type="clinical.patient.created",
            outcome="success",
            actor_type="user",
            actor_id="user_integ_03",
            tenant_id="tenant_audit_integ",
            patient_id=test_id,
        )
        ev_id = ev.id

        # Simulate exception and rollback
        db.rollback()

        # Both patient and audit event must NOT exist
        assert db.query(Patient).filter(Patient.id == test_id).first() is None
        assert db.query(AuditEvent).filter(AuditEvent.id == ev_id).first() is None
    finally:
        db.close()


def test_audit_security_event_isolated_commit():
    """Verify that security audit events commit independently and survive caller rollback/exceptions."""
    db = SessionLocal()
    test_username = f"sec_user_{uuid.uuid4().hex[:6]}"
    try:
        # Start a caller transaction and add dummy object
        p = Patient(
            id=f"pat_tmp_{uuid.uuid4().hex[:6]}",
            tenant_id="tenant_audit_integ",
            name="Temp Patient",
            age=20,
            gender="F",
        )
        db.add(p)

        # Record security event (uses isolated session)
        ev = record_security_event(
            event_type="auth.login.failure",
            outcome="failure",
            actor_type="anonymous",
            metadata={"attempted_username": test_username},
        )
        assert ev is not None
        ev_id = ev

        # Caller encounters error and rolls back its transaction
        db.rollback()

        # The temporary patient was rolled back
        assert db.query(Patient).filter(Patient.id == p.id).first() is None

        # The security audit event PERSISTS in the DB
        persisted_ev = db.query(AuditEvent).filter(AuditEvent.id == ev_id).first()
        assert persisted_ev is not None
        assert persisted_ev.event_type == "auth.login.failure"
        assert persisted_ev.metadata_json.get("attempted_username") == test_username
    finally:
        db.close()


def test_audit_historical_survival_after_patient_deletion():
    """
    Verify that an audit record survives deletion of the referenced Patient.
    Audit records use unconstrained historical patient_id strings, not CASCADE foreign keys.
    """
    db = SessionLocal()
    test_pid = f"pat_hist_{uuid.uuid4().hex[:6]}"
    try:
        patient = Patient(
            id=test_pid,
            tenant_id="tenant_audit_integ",
            name="Historical Patient",
            age=65,
            gender="M",
        )
        db.add(patient)
        ev = record_audit_event(
            db=db,
            event_type="clinical.patient.created",
            outcome="success",
            actor_type="user",
            actor_id="user_integ_04",
            tenant_id="tenant_audit_integ",
            patient_id=test_pid,
        )
        db.commit()
        ev_id = ev.id

        # Delete the patient record
        db.delete(patient)
        db.commit()

        # Patient is gone
        assert db.query(Patient).filter(Patient.id == test_pid).first() is None

        # Audit event MUST still exist and retain the patient_id
        surviving_ev = db.query(AuditEvent).filter(AuditEvent.id == ev_id).first()
        assert surviving_ev is not None
        assert surviving_ev.patient_id == test_pid
    finally:
        db.close()
