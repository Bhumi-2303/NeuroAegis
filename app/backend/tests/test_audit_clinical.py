from __future__ import annotations
import datetime
import io
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.auth import create_access_token, get_password_hash
from app.db.database import Base, SessionLocal, engine, ensure_schema_compatibility
from app.db.models import AuditEvent, Patient, PredictionJob, Tenant, User
from app.main import app
from app.services.job_recovery import reap_stale_jobs
from app.services.job_service import run_prediction_pipeline
from tests.edf_fixture import write_synthetic_edf

CLIENT_CLINICAL = TestClient(app)

TENANT_CLIN = "tenant_audit_clin_01"
USER_ADMIN = "user_clin_admin_01"
USER_CLIN = "user_clin_doctor_01"


@pytest.fixture(scope="module", autouse=True)
def setup_clinical_audit_db():
    Base.metadata.create_all(bind=engine)
    ensure_schema_compatibility(engine)
    db = SessionLocal()
    try:
        if not db.query(Tenant).filter(Tenant.id == TENANT_CLIN).first():
            db.add(Tenant(id=TENANT_CLIN, name="Clinical Audit Clinic", slug=TENANT_CLIN, is_active=True))

        for u_id, u_name, role in [
            (USER_ADMIN, "clin_admin_01", "admin"),
            (USER_CLIN, "clin_doctor_01", "clinician"),
        ]:
            if not db.query(User).filter(User.id == u_id).first():
                db.add(
                    User(
                        id=u_id,
                        tenant_id=TENANT_CLIN,
                        username=u_name,
                        hashed_password=get_password_hash("ClinPass123!"),
                        role=role,
                        is_active=True,
                    )
                )
        db.commit()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def ensure_models_loaded():
    """Ensure prediction router models are loaded for testing API v2 dispatch."""
    from app.services.prediction.prediction_router import prediction_router
    if not prediction_router.is_loaded:
        prediction_router.load_all_models()


def make_token(user_id: str, role: str, tenant_id: str) -> str:
    user = User(
        id=user_id,
        username=f"u_{user_id}",
        tenant_id=tenant_id,
        role=role,
        token_version=1,
    )
    return create_access_token(user)


def make_synthetic_chbmit_edf_bytes() -> bytes:
    """Constructs a minimal 23-channel CHB-MIT-compatible EDF in memory."""
    ch_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ",
        "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
    ]
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        write_synthetic_edf(tmp_path, channel_names=ch_names, sampling_rate=256, duration_seconds=1)
        return tmp_path.read_bytes()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_audit_patient_created():
    """Verify clinical.patient.created event is recorded upon patient registration."""
    token = make_token(USER_CLIN, "clinician", TENANT_CLIN)
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "name": "Audit Patient Alpha",
        "age": 52,
        "gender": "F",
        "weight": 65.0,
        "height": 165.0,
    }
    resp = CLIENT_CLINICAL.post("/api/v1/patients/", json=payload, headers=headers)
    assert resp.status_code == 201
    created_id = resp.json()["id"]

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "clinical.patient.created",
                AuditEvent.resource_id == created_id,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "success"
        assert event.actor_id == USER_CLIN
        assert event.tenant_id == TENANT_CLIN
        assert event.patient_id == created_id
        # Invariant: PHI (patient name, age) not leaked in metadata
        meta = str(event.metadata_json or {})
        assert "Audit Patient Alpha" not in meta
    finally:
        db.close()


def test_audit_patient_viewed_single():
    """Verify clinical.patient.viewed is recorded when a specific patient record is fetched."""
    token = make_token(USER_CLIN, "clinician", TENANT_CLIN)
    headers = {"Authorization": f"Bearer {token}"}

    payload = {"name": "Single View Patient", "age": 30, "gender": "M", "weight": 70.0, "height": 175.0}
    c_resp = CLIENT_CLINICAL.post("/api/v1/patients/", json=payload, headers=headers)
    assert c_resp.status_code == 201
    p_id = c_resp.json()["id"]

    # View the patient
    v_resp = CLIENT_CLINICAL.get(f"/api/v1/patients/{p_id}", headers=headers)
    assert v_resp.status_code == 200

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "clinical.patient.viewed",
                AuditEvent.patient_id == p_id,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "success"
        assert event.actor_id == USER_CLIN
        assert event.tenant_id == TENANT_CLIN
    finally:
        db.close()


def test_audit_patient_viewed_list():
    """Verify clinical.patient.viewed is recorded with count metadata on list fetch."""
    token = make_token(USER_CLIN, "clinician", TENANT_CLIN)
    headers = {"Authorization": f"Bearer {token}"}

    resp = CLIENT_CLINICAL.get("/api/v1/patients/", headers=headers)
    assert resp.status_code == 200
    count = len(resp.json())

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "clinical.patient.viewed",
                AuditEvent.resource_type == "patient",
                AuditEvent.resource_id == None,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "success"
        assert event.tenant_id == TENANT_CLIN
        meta = event.metadata_json or {}
        assert meta.get("count") == count
    finally:
        db.close()


def test_audit_patient_deleted():
    """Verify clinical.patient.deleted is recorded atomically before deletion and survives after patient is deleted."""
    token_clin = make_token(USER_CLIN, "clinician", TENANT_CLIN)
    c_resp = CLIENT_CLINICAL.post(
        "/api/v1/patients/",
        json={"name": "To Delete Patient", "age": 40, "gender": "F", "weight": 60.0, "height": 160.0},
        headers={"Authorization": f"Bearer {token_clin}"},
    )
    assert c_resp.status_code == 201
    p_id = c_resp.json()["id"]

    # Delete patient as admin
    token_admin = make_token(USER_ADMIN, "admin", TENANT_CLIN)
    d_resp = CLIENT_CLINICAL.delete(
        f"/api/v1/data/patient/{p_id}",
        headers={"Authorization": f"Bearer {token_admin}"},
    )
    assert d_resp.status_code == 200

    db = SessionLocal()
    try:
        # Patient row must be gone
        assert db.query(Patient).filter(Patient.id == p_id).first() is None

        # Audit event MUST still exist!
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "clinical.patient.deleted",
                AuditEvent.patient_id == p_id,
            )
            .order_by(AuditEvent.occurred_at.desc())
            .first()
        )
        assert event is not None
        assert event.outcome == "success"
        assert event.actor_id == USER_ADMIN
        assert event.tenant_id == TENANT_CLIN
        assert event.patient_id == p_id
    finally:
        db.close()


def test_audit_job_created():
    """Verify clinical.job.created is recorded on prediction dispatch."""
    token = make_token(USER_CLIN, "clinician", TENANT_CLIN)
    headers = {"Authorization": f"Bearer {token}"}

    # Create patient
    p_resp = CLIENT_CLINICAL.post(
        "/api/v1/patients/",
        json={"name": "Job Patient", "age": 60, "gender": "M", "weight": 75.0, "height": 180.0},
        headers=headers,
    )
    assert p_resp.status_code == 201
    patient_id = p_resp.json()["id"]

    edf_bytes = make_synthetic_chbmit_edf_bytes()
    files = {"file": ("chb01_01.edf", io.BytesIO(edf_bytes), "application/octet-stream")}
    data = {
        "patient_id": patient_id,
        "name": "Job Patient",
        "age": 60,
        "gender": "female",
        "weight": 75.0,
        "height": 180.0,
        "medical_history": "{}",
        "vital_signs": "{}",
        "execution_mode": "sync",
    }

    with patch("app.api.v2.predict.run_prediction_pipeline"):
        resp = CLIENT_CLINICAL.post("/api/v2/predict/", data=data, files=files, headers=headers)
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "clinical.job.created",
                AuditEvent.job_id == job_id,
            )
            .first()
        )
        assert event is not None
        assert event.outcome == "success"
        assert event.patient_id == patient_id
        assert event.tenant_id == TENANT_CLIN
        meta = event.metadata_json or {}
        assert meta.get("dataset") == "chbmit"
    finally:
        db.close()


def test_audit_job_report_viewed():
    """Verify clinical.job.report_viewed is recorded when report endpoint is hit."""
    token = make_token(USER_CLIN, "clinician", TENANT_CLIN)
    headers = {"Authorization": f"Bearer {token}"}

    # Seed completed job
    db = SessionLocal()
    try:
        job_id = f"job_report_{uuid.uuid4().hex[:8]}"
        p_id = f"pat_rep_{uuid.uuid4().hex[:8]}"
        db.add(
            Patient(
                id=p_id,
                tenant_id=TENANT_CLIN,
                name="Report Patient",
                age=25,
                gender="M",
                weight=70.0,
                height=175.0,
            )
        )
        db.add(
            PredictionJob(
                id=job_id,
                tenant_id=TENANT_CLIN,
                patient_id=p_id,
                status="completed",
                prediction_label="Seizure",
                probability_seizure=0.95,
            )
        )
        db.commit()
    finally:
        db.close()

    resp = CLIENT_CLINICAL.get(f"/api/v2/report/{job_id}", headers=headers)
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "clinical.job.report_viewed",
                AuditEvent.job_id == job_id,
            )
            .first()
        )
        assert event is not None
        assert event.outcome == "success"
        assert event.actor_id == USER_CLIN
        assert event.patient_id == p_id
    finally:
        db.close()


def test_audit_prediction_completed():
    """Verify clinical.prediction.completed is recorded by worker upon pipeline success."""
    db = SessionLocal()
    try:
        job_id = f"job_comp_{uuid.uuid4().hex[:8]}"
        p_id = f"pat_comp_{uuid.uuid4().hex[:8]}"
        db.add(
            Patient(
                id=p_id,
                tenant_id=TENANT_CLIN,
                name="Complete Patient",
                age=45,
                gender="F",
                weight=60.0,
                height=165.0,
            )
        )
        job = PredictionJob(
            id=job_id,
            tenant_id=TENANT_CLIN,
            patient_id=p_id,
            status="running",
            worker_id="worker-test-node:123:abc",
        )
        db.add(job)
        db.commit()

        # Run pipeline with mocked inference
        with patch("app.services.job_service._run_inference") as mock_infer:
            mock_infer.return_value = {
                "label": "Non-Seizure",
                "prob_seizure": 0.12,
                "band": "High",
                "explanation": {"shap_values": []},
            }
            import asyncio
            asyncio.run(
                run_prediction_pipeline(
                    job_id=job_id,
                    eeg_data=np.zeros((19, 500), dtype=np.float32),
                    channel_names=[f"EEG-{i}" for i in range(19)],
                    fs=250.0,
                    expected_worker_id="worker-test-node:123:abc",
                )
            )

        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "clinical.prediction.completed",
                AuditEvent.job_id == job_id,
            )
            .first()
        )
        assert event is not None
        assert event.outcome == "success"
        assert event.actor_type == "worker"
        assert event.patient_id == p_id
        assert event.tenant_id == TENANT_CLIN
    finally:
        db.close()


def test_audit_prediction_failed():
    """Verify clinical.prediction.failed is recorded with sanitized error category upon pipeline failure."""
    db = SessionLocal()
    try:
        job_id = f"job_fail_{uuid.uuid4().hex[:8]}"
        p_id = f"pat_fail_{uuid.uuid4().hex[:8]}"
        db.add(
            Patient(
                id=p_id,
                tenant_id=TENANT_CLIN,
                name="Fail Patient",
                age=45,
                gender="F",
                weight=65.0,
                height=165.0,
            )
        )
        job = PredictionJob(
            id=job_id,
            tenant_id=TENANT_CLIN,
            patient_id=p_id,
            status="running",
            worker_id="worker-test-node:123:abc",
        )
        db.add(job)
        db.commit()

        with patch("app.services.job_service._run_inference", side_effect=RuntimeError("Internal model matrix failure!")):
            import asyncio
            asyncio.run(
                run_prediction_pipeline(
                    job_id=job_id,
                    eeg_data=np.zeros((19, 500), dtype=np.float32),
                    channel_names=[f"EEG-{i}" for i in range(19)],
                    fs=250.0,
                    expected_worker_id="worker-test-node:123:abc",
                )
            )

        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "clinical.prediction.failed",
                AuditEvent.job_id == job_id,
            )
            .first()
        )
        assert event is not None
        assert event.outcome == "failure"
        assert event.actor_type == "worker"
        assert event.patient_id == p_id
        assert event.tenant_id == TENANT_CLIN
        # Error category must be sanitized (never raw stack or internal trace)
        assert event.error_category is not None
        assert "Internal model matrix" not in str(event.error_category)
    finally:
        db.close()


def test_audit_worker_lease_expired():
    """Verify reaper creates system.worker.lease_expired event when reclaiming a dead worker job."""
    db = SessionLocal()
    try:
        job_id = f"job_reap_{uuid.uuid4().hex[:8]}"
        p_id = f"pat_reap_{uuid.uuid4().hex[:8]}"
        db.add(
            Patient(
                id=p_id,
                tenant_id=TENANT_CLIN,
                name="Reap Patient",
                age=45,
                gender="F",
                weight=60.0,
                height=160.0,
            )
        )
        stale_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120)
        job = PredictionJob(
            id=job_id,
            tenant_id=TENANT_CLIN,
            patient_id=p_id,
            status="running",
            worker_id="dead-worker:999:dead",
            heartbeat_at=stale_time,
            lease_expires_at=stale_time,
            created_at=stale_time,
        )
        db.add(job)
        db.commit()

        reaped_count = reap_stale_jobs(db)
        assert len(reaped_count) >= 1

        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.event_type == "system.worker.lease_expired",
                AuditEvent.job_id == job_id,
            )
            .first()
        )
        assert event is not None
        assert event.outcome == "failure"
        assert event.actor_type == "system"
        assert event.patient_id == p_id
        assert event.tenant_id == TENANT_CLIN
    finally:
        db.close()
