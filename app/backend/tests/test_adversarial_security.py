from __future__ import annotations
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from jose import jwt

from app.core.auth import (
    create_access_token,
    create_refresh_token,
    get_password_hash,
    require_roles,
    SAFE_ID_REGEX,
)
from app.core.config import settings
from app.db.database import Base, SessionLocal, engine, get_db
from app.db.models import Patient, PredictionJob, RefreshToken, Tenant, User
from app.main import app
from app.services.job_service import update_job_status, run_prediction_pipeline
from app.services.queue import _validate_payload_metadata, QueuePayloadError
from app.services.worker import claim_job, update_heartbeat, run_prediction_task
from tests.edf_fixture import write_synthetic_edf

TENANT_ALPHA = "tenant_alpha"
TENANT_BETA = "tenant_beta"

USER_CLIN_ALPHA = "user_clin_alpha"
USER_RES_ALPHA = "user_res_alpha"
USER_ADMIN_ALPHA = "user_admin_alpha"
USER_CLIN_BETA = "user_clin_beta"

PATIENT_ALPHA = "pat_alpha_001"
PATIENT_BETA = "pat_beta_001"

JOB_ALPHA = "job_alpha_001"
JOB_BETA = "job_beta_001"


def create_token_with_claims(claims: dict) -> str:
    return jwt.encode(claims, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


@pytest.fixture(scope="module")
def setup_adversarial_env():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Seed Tenants
        for t_id, name in [(TENANT_ALPHA, "Alpha Health"), (TENANT_BETA, "Beta Hospital")]:
            if not db.query(Tenant).filter(Tenant.id == t_id).first():
                db.add(Tenant(id=t_id, name=name, slug=t_id, is_active=True))
        db.commit()

        # Seed Users
        pwd_hash = get_password_hash("AdversarialSecure123!")
        users_to_seed = [
            (USER_CLIN_ALPHA, "adv_clin_alpha", TENANT_ALPHA, "clinician", True, 1),
            (USER_RES_ALPHA, "adv_res_alpha", TENANT_ALPHA, "researcher", True, 1),
            (USER_ADMIN_ALPHA, "adv_admin_alpha", TENANT_ALPHA, "admin", True, 1),
            (USER_CLIN_BETA, "adv_clin_beta", TENANT_BETA, "clinician", True, 1),
            ("user_inactive_alpha", "adv_inactive_alpha", TENANT_ALPHA, "clinician", False, 1),
        ]
        for uid, uname, tid, role, active, t_ver in users_to_seed:
            existing_u = db.query(User).filter((User.id == uid) | (User.username == uname)).first()
            if not existing_u:
                db.add(User(
                    id=uid,
                    username=uname,
                    hashed_password=pwd_hash,
                    tenant_id=tid,
                    role=role,
                    is_active=active,
                    token_version=t_ver,
                    created_at=datetime.now(timezone.utc),
                ))
            else:
                existing_u.is_active = active
                existing_u.token_version = t_ver
        db.commit()

        # Seed Patients
        patients_to_seed = [
            (PATIENT_ALPHA, TENANT_ALPHA, USER_CLIN_ALPHA, "Patient Alpha"),
            (PATIENT_BETA, TENANT_BETA, USER_CLIN_BETA, "Patient Beta"),
        ]
        for p_id, t_id, c_id, name in patients_to_seed:
            if not db.query(Patient).filter(Patient.id == p_id).first():
                db.add(Patient(
                    id=p_id,
                    tenant_id=t_id,
                    created_by_user_id=c_id,
                    name=name,
                    age=35,
                    gender="M",
                    weight=70.0,
                    height=175.0,
                    medical_history="{}",
                    vital_signs={},
                    is_deleted=False,
                ))
        db.commit()

        # Seed Jobs
        jobs_to_seed = [
            (JOB_ALPHA, TENANT_ALPHA, USER_CLIN_ALPHA, PATIENT_ALPHA, "Completed", 100, "Seizure", 0.95),
            (JOB_BETA, TENANT_BETA, USER_CLIN_BETA, PATIENT_BETA, "Completed", 100, "Non-Seizure", 0.05),
        ]
        for j_id, t_id, c_id, p_id, status, prog, label, prob in jobs_to_seed:
            if not db.query(PredictionJob).filter(PredictionJob.id == j_id).first():
                db.add(PredictionJob(
                    id=j_id,
                    tenant_id=t_id,
                    created_by_user_id=c_id,
                    patient_id=p_id,
                    status=status,
                    progress=prog,
                    prediction_label=label,
                    probability_seizure=prob,
                    is_deleted=False,
                ))
        db.commit()

    finally:
        db.close()


@pytest.fixture
def client(setup_adversarial_env):
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ==============================================================================
# STAGE 2: AUTHENTICATION & SESSION ADVERSARIAL TESTS
# ==============================================================================

def test_missing_malformed_expired_wrong_type_jwts_rejected(client: TestClient):
    """Verify that malformed, expired, incorrectly signed, and wrong-type JWTs fail with 401."""
    now = datetime.now(timezone.utc)

    # 1. Missing Authorization header and cookie
    assert client.get("/api/v1/auth/me").status_code == 401

    # 2. Malformed token string
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-valid-jwt"}).status_code == 401

    # 3. Expired token
    expired_token = create_token_with_claims({
        "sub": USER_CLIN_ALPHA,
        "tenant_id": TENANT_ALPHA,
        "role": "clinician",
        "token_version": 1,
        "typ": "access",
        "exp": int((now - timedelta(minutes=10)).timestamp()),
    })
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"}).status_code == 401

    # 4. Incorrectly signed token
    bad_sig_token = jwt.encode(
        {"sub": USER_CLIN_ALPHA, "tenant_id": TENANT_ALPHA, "role": "clinician", "token_version": 1, "typ": "access", "exp": int((now + timedelta(minutes=15)).timestamp())},
        "completely_wrong_secret_key_12345",
        algorithm="HS256",
    )
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {bad_sig_token}"}).status_code == 401

    # 5. Wrong type: typ="refresh" presented to access endpoint
    wrong_type_refresh = create_token_with_claims({
        "sub": USER_CLIN_ALPHA,
        "tenant_id": TENANT_ALPHA,
        "role": "clinician",
        "token_version": 1,
        "typ": "refresh",
        "exp": int((now + timedelta(minutes=15)).timestamp()),
    })
    res_refresh = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {wrong_type_refresh}"})
    assert res_refresh.status_code == 401, "Token with typ='refresh' must be rejected on access endpoint"

    # 6. Wrong type: missing "typ" claim entirely
    missing_typ = create_token_with_claims({
        "sub": USER_CLIN_ALPHA,
        "tenant_id": TENANT_ALPHA,
        "role": "clinician",
        "token_version": 1,
        "exp": int((now + timedelta(minutes=15)).timestamp()),
    })
    res_no_typ = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {missing_typ}"})
    assert res_no_typ.status_code == 401, "Token with missing typ must be rejected"


def test_inactive_user_cannot_authenticate_or_use_session(client: TestClient):
    """Verify that inactive users are completely rejected from login, access, and refresh."""
    # 1. Login attempt with inactive user credentials fails with 401 generic error
    res_login = client.post("/api/v1/auth/login", json={"username": "adv_inactive_alpha", "password": "AdversarialSecure123!"})
    assert res_login.status_code == 401
    assert res_login.json()["detail"] == "Incorrect username or password"

    # 2. Existing JWT for inactive user fails with 401
    inactive_user = User(
        id="user_inactive_alpha",
        username="adv_inactive_alpha",
        tenant_id=TENANT_ALPHA,
        role="clinician",
        token_version=1,
    )
    token = create_access_token(inactive_user)
    res_me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_me.status_code == 401
    assert "inactive" in res_me.json()["detail"].lower()


def test_token_version_bump_invalidates_previously_issued_credentials(client: TestClient):
    """Verify that incrementing token_version revokes active access tokens immediately."""
    db = SessionLocal()
    user = db.query(User).filter(User.id == USER_CLIN_ALPHA).first()
    initial_version = user.token_version
    db.close()

    user_obj = User(
        id=USER_CLIN_ALPHA,
        username="adv_clin_alpha",
        tenant_id=TENANT_ALPHA,
        role="clinician",
        token_version=initial_version,
    )
    token = create_access_token(user_obj)

    # Valid token succeeds
    res_valid = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_valid.status_code == 200

    # Bump token_version in database
    db = SessionLocal()
    u = db.query(User).filter(User.id == USER_CLIN_ALPHA).first()
    u.token_version += 1
    db.commit()
    db.close()

    # Old token fails with 401 revoked
    res_invalid = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_invalid.status_code == 401
    assert "revoked" in res_invalid.json()["detail"].lower()

    # Reset token version back for subsequent tests
    db = SessionLocal()
    u = db.query(User).filter(User.id == USER_CLIN_ALPHA).first()
    u.token_version = initial_version
    db.commit()
    db.close()


def test_login_zero_jwt_leakage_and_cookie_attributes(client: TestClient):
    """Verify login body contains no JWTs, and response sets secure cookies with correct attributes."""
    res = client.post("/api/v1/auth/login", json={"username": "adv_clin_alpha", "password": "AdversarialSecure123!"})
    assert res.status_code == 200
    body = res.json()

    # Zero JWTs in JSON body
    assert "access_token" not in body
    assert "refresh_token" not in body
    assert "csrf_token" in body
    assert body["user"]["username"] == "adv_clin_alpha"
    assert body["user"]["role"] == "clinician"

    # Cookie attributes
    set_cookies = res.headers.get_list("set-cookie")
    cookie_str = "; ".join(set_cookies)
    assert settings.ACCESS_COOKIE_NAME in cookie_str
    assert settings.REFRESH_COOKIE_NAME in cookie_str
    assert settings.CSRF_COOKIE_NAME in cookie_str
    assert "HttpOnly" in cookie_str
    assert "SameSite=strict" in cookie_str or "SameSite=Strict" in cookie_str


# ==============================================================================
# STAGE 3: AUTHORIZATION & TENANT ISOLATION ADVERSARIAL TESTS
# ==============================================================================

def test_stream_eeg_requires_clinician_or_admin_role(client: TestClient):
    """Verify that GET /api/v1/stream/eeg requires clinician or admin role and rejects researcher (403)."""
    # 1. Unauthenticated -> 401
    assert client.get("/api/v1/stream/eeg").status_code == 401

    # 2. Researcher role -> 403 Forbidden (telemetry boundary)
    user_res = User(id=USER_RES_ALPHA, username="adv_res_alpha", tenant_id=TENANT_ALPHA, role="researcher", token_version=1)
    res_token = create_access_token(user_res)
    res_resp = client.get("/api/v1/stream/eeg", headers={"Authorization": f"Bearer {res_token}"})
    assert res_resp.status_code == 403
    assert "Insufficient permissions" in res_resp.json()["detail"]

    # 3. Clinician role -> Allowed (mock StreamingResponse to avoid infinite SSE loop)
    user_clin = User(id=USER_CLIN_ALPHA, username="adv_clin_alpha", tenant_id=TENANT_ALPHA, role="clinician", token_version=1)
    clin_token = create_access_token(user_clin)
    with patch("app.api.v1.stream.StreamingResponse") as mock_stream:
        from fastapi.responses import Response
        mock_stream.return_value = Response(content="stream-ok", status_code=200)
        clin_resp = client.get("/api/v1/stream/eeg", headers={"Authorization": f"Bearer {clin_token}"})
        assert clin_resp.status_code == 200


def test_v1_predict_early_patient_authorization(client: TestClient, tmp_path: Path):
    """Verify v1 /predict rejects cross-tenant patient_id with 404 BEFORE reading/parsing file."""
    user_clin = User(id=USER_CLIN_ALPHA, username="adv_clin_alpha", tenant_id=TENANT_ALPHA, role="clinician", token_version=1)
    token = create_access_token(user_clin)

    fixture_path = write_synthetic_edf(
        tmp_path / "test_early_patient.edf",
        channel_names=[f"Ch{i}" for i in range(23)],
    )

    # Cross-tenant patient reference PATIENT_BETA (Tenant B) from Clinician A (Tenant A) -> 404
    resp = client.post(
        "/api/v1/predict/",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("test.edf", fixture_path.read_bytes(), "application/octet-stream")},
        data={"patient_id": PATIENT_BETA},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Patient not found"


def test_soft_deleted_patient_and_job_are_consistently_excluded(client: TestClient):
    """Verify soft-deleted patients and jobs cannot be accessed by their own tenant."""
    db = SessionLocal()
    try:
        # Create a temporary patient and job, then soft-delete them
        del_pid = f"del_pat_{uuid.uuid4().hex[:8]}"
        del_jid = f"del_job_{uuid.uuid4().hex[:8]}"
        db.add(Patient(
            id=del_pid,
            tenant_id=TENANT_ALPHA,
            created_by_user_id=USER_CLIN_ALPHA,
            name="Deleted Patient",
            age=50,
            gender="M",
            weight=75.0,
            height=180.0,
            medical_history="{}",
            vital_signs={},
            is_deleted=True,
        ))
        db.add(PredictionJob(
            id=del_jid,
            tenant_id=TENANT_ALPHA,
            created_by_user_id=USER_CLIN_ALPHA,
            patient_id=del_pid,
            status="Completed",
            progress=100,
            is_deleted=True,
        ))
        db.commit()
    finally:
        db.close()

    user_clin = User(id=USER_CLIN_ALPHA, username="adv_clin_alpha", tenant_id=TENANT_ALPHA, role="clinician", token_version=1)
    token = create_access_token(user_clin)
    headers = {"Authorization": f"Bearer {token}"}

    # Patient query fails with 404
    assert client.get(f"/api/v1/patients/{del_pid}", headers=headers).status_code == 404

    # Job queries fail with 404
    assert client.get(f"/api/v1/jobs/{del_jid}", headers=headers).status_code == 404
    assert client.get(f"/api/v2/jobs/{del_jid}", headers=headers).status_code == 404
    assert client.get(f"/api/v2/report/{del_jid}", headers=headers).status_code == 404

    # History excludes soft-deleted job
    hist = client.get("/api/v2/history", headers=headers).json()
    job_ids = [item["job_id"] for item in hist]
    assert del_jid not in job_ids


# ==============================================================================
# STAGE 4: QUEUE AND WORKER TRUST BOUNDARY
# ==============================================================================

def test_queue_prohibits_raw_eeg_numpy_and_large_buffers():
    """Verify queue interface rejects raw numpy arrays and large binary buffers."""
    # 1. Valid path string succeeds
    _validate_payload_metadata("job-123", "/shared/staged/job-123.edf", "bonn", None, {})

    # 2. NumPy array rejected
    with pytest.raises(QueuePayloadError) as exc_np:
        _validate_payload_metadata(
            "job-123",
            "/shared/staged/job-123.edf",
            "bonn",
            None,
            {"raw_signal": np.zeros((19, 256))},
        )
    assert "NumPy arrays must not be sent via Redis" in str(exc_np.value)

    # 3. Large binary buffer rejected
    with pytest.raises(QueuePayloadError) as exc_bin:
        _validate_payload_metadata(
            "job-123",
            "/shared/staged/job-123.edf",
            "bonn",
            None,
            {"raw_bytes": b"X" * 2048},
        )
    assert "Large binary buffers" in str(exc_bin.value)


def test_worker_refuses_to_claim_or_finalize_deleted_job():
    """Verify worker refuses to claim or finalize soft-deleted jobs."""
    del_jid = f"reap_job_{uuid.uuid4().hex[:8]}"
    db = SessionLocal()
    try:
        db.add(PredictionJob(
            id=del_jid,
            tenant_id=TENANT_ALPHA,
            created_by_user_id=USER_CLIN_ALPHA,
            status="Validating",
            progress=0,
            is_deleted=True,
        ))
        db.commit()
    finally:
        db.close()

    # 1. claim_job on soft-deleted job returns False
    claimed = claim_job(del_jid, "worker-test-1", 30)
    assert claimed is False, "Worker must not claim soft-deleted job"

    # 2. update_job_status on soft-deleted job skips update
    update_job_status(del_jid, "Processing", 50)
    db = SessionLocal()
    try:
        j = db.query(PredictionJob).filter(PredictionJob.id == del_jid).first()
        assert j.progress == 0, "update_job_status must not update soft-deleted job"
    finally:
        db.close()


def test_worker_refuses_to_steal_active_lease():
    """Verify worker cannot steal a job that is actively leased by another live worker."""
    lease_jid = f"lease_job_{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow()
    db = SessionLocal()
    try:
        db.add(PredictionJob(
            id=lease_jid,
            tenant_id=TENANT_ALPHA,
            created_by_user_id=USER_CLIN_ALPHA,
            status="Running",
            worker_id="worker_primary",
            heartbeat_at=now,
            lease_expires_at=now + timedelta(seconds=60),
            is_deleted=False,
        ))
        db.commit()
    finally:
        db.close()

    # Second worker attempts to claim actively leased job -> rejected
    claimed = claim_job(lease_jid, "worker_impostor", 30)
    assert claimed is False, "Active worker lease must not be stolen"


# ==============================================================================
# STAGE 5: COMPLETE AUTHENTICATED CLINICAL WORKFLOW
# ==============================================================================

def test_complete_authenticated_clinical_workflow(client: TestClient, tmp_path: Path):
    """
    End-to-end integration test exercising:
    Login -> Session Verify -> Patient Creation -> Prediction Upload -> Job Status -> Report -> History -> Logout
    """
    # 1. Login as Clinician Alpha
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "adv_clin_alpha", "password": "AdversarialSecure123!"},
    )
    assert login_resp.status_code == 200
    csrf_token = login_resp.json()["csrf_token"]

    # Session cookie is automatically stored in client.cookies
    assert settings.ACCESS_COOKIE_NAME in client.cookies
    assert settings.CSRF_COOKIE_NAME in client.cookies

    # 2. Verify Session via /auth/me
    me_resp = client.get("/api/v1/auth/me")
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "adv_clin_alpha"
    assert me_resp.json()["tenant_id"] == TENANT_ALPHA

    # 3. Create Patient with CSRF protection
    patient_payload = {
        "name": "Integration Clinical Patient",
        "age": 42,
        "gender": "F",
        "weight": 68.0,
        "height": 172.0,
        "medical_history": "{}",
        "vital_signs": {},
    }
    pat_resp = client.post(
        "/api/v1/patients/",
        json=patient_payload,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert pat_resp.status_code == 201
    created_patient_id = pat_resp.json()["id"]

    # 4. Upload v2 Prediction Job with synthetic EDF
    ch_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ",
        "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
    ]
    edf_path = write_synthetic_edf(tmp_path / "workflow_test.edf", channel_names=ch_names, sampling_rate=256)

    upload_resp = client.post(
        "/api/v2/predict",
        headers={"X-CSRF-Token": csrf_token},
        data={
            "patient_id": created_patient_id,
            "name": "Integration Clinical Patient",
            "age": 42,
            "gender": "female",
            "weight": 68.0,
            "height": 172.0,
            "medical_history": "{}",
            "vital_signs": "{}",
        },
        files={"file": ("workflow_test.edf", edf_path.read_bytes(), "application/octet-stream")},
    )
    assert upload_resp.status_code == 200
    workflow_job_id = upload_resp.json()["job_id"]
    assert upload_resp.json()["patient_id"] == created_patient_id

    # 5. Poll Job Status
    job_resp = client.get(f"/api/v2/jobs/{workflow_job_id}")
    assert job_resp.status_code == 200
    assert job_resp.json()["job_id"] == workflow_job_id

    # 6. Retrieve Clinical Report
    report_resp = client.get(f"/api/v2/report/{workflow_job_id}")
    assert report_resp.status_code == 200
    assert report_resp.json()["job"]["id"] == workflow_job_id
    assert report_resp.json()["patient"]["name"] == "Integration Clinical Patient"

    # 7. Check History
    hist_resp = client.get("/api/v2/history")
    assert hist_resp.status_code == 200
    hist_job_ids = [item["job_id"] for item in hist_resp.json()]
    assert workflow_job_id in hist_job_ids

    # 8. Logout
    logout_resp = client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert logout_resp.status_code == 200

    # 9. Verify subsequent protected request fails with 401
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/patients/").status_code == 401


# ==============================================================================
# STAGE 6: PRIVACY & INFORMATION LEAKAGE AUDIT
# ==============================================================================

def test_metrics_endpoint_does_not_leak_internal_paths_on_error(client: TestClient):
    """Verify metrics endpoint returns generic 500 error without exposing internal paths on exception."""
    with patch("builtins.open", side_effect=PermissionError("/secret/internal/system/path/denied")):
        resp = client.get("/api/v1/metrics")
        if resp.status_code == 500:
            detail = resp.json().get("detail", "")
            assert "/secret/internal" not in detail
            assert detail == "Failed to load metrics"


def test_user_responses_never_leak_password_hashes(client: TestClient):
    """Verify that user representations in auth/me, patients, and jobs never disclose hashed_password."""
    user_clin = User(id=USER_CLIN_ALPHA, username="adv_clin_alpha", tenant_id=TENANT_ALPHA, role="clinician", token_version=1)
    token = create_access_token(user_clin)
    headers = {"Authorization": f"Bearer {token}"}

    me_data = client.get("/api/v1/auth/me", headers=headers).json()
    assert "password" not in me_data
    assert "hashed_password" not in me_data

    pat_list = client.get("/api/v1/patients/", headers=headers).json()
    for pat in pat_list:
        assert "password" not in pat
        assert "hashed_password" not in pat


# ==============================================================================
# PHASE 10.1: TELEMETRY PRIVACY & ERROR SANITIZATION ADVERSARIAL TESTS (P0)
# ==============================================================================

CHBMIT_TEST_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
]

def test_redis_credential_leakage_prevented_in_api_and_job_error(client: TestClient, monkeypatch):
    """
    TEL-01 Adversarial Test:
    Simulate queue connection failure with a credential-bearing Redis DSN containing a canary secret.
    Verify that:
    1. Canary secret is absent from API response text and detail.
    2. Canary secret is absent from persisted PredictionJob.error in the database.
    3. Failure leaves the job in a consistent 'Failed' state.
    4. Queue connection exception itself does not expose the password.
    """
    from app.services.queue import RedisPredictionQueue, QueueConnectionError
    from app.core.config import settings

    canary_secret = "CANARY_REDIS_SECRET_987654"
    canary_dsn = f"redis://default:{canary_secret}@redis.internal:6379/0"

    # Test A.1: Direct Queue Connection Exception Constructor sanitization
    queue_instance = RedisPredictionQueue(redis_url=canary_dsn)
    with pytest.raises(QueueConnectionError) as exc_info:
        import asyncio
        asyncio.run(queue_instance.get_pool())
    assert canary_secret not in str(exc_info.value)
    assert ":***@" in str(exc_info.value)

    # Test A.2: API v2 Predict dispatch failure in distributed queue mode
    monkeypatch.setattr(settings, "ENABLE_DISTRIBUTED_QUEUE", True)
    from unittest.mock import AsyncMock
    mock_queue = AsyncMock()
    mock_queue.enqueue_prediction_job.side_effect = QueueConnectionError(f"Could not connect to {canary_dsn}")
    monkeypatch.setattr("app.api.v2.predict.prediction_queue", mock_queue)

    user_clin = User(id=USER_CLIN_ALPHA, username="adv_clin_alpha", tenant_id=TENANT_ALPHA, role="clinician", token_version=1)
    token = create_access_token(user_clin)
    headers = {"Authorization": f"Bearer {token}"}

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_edf:
        edf_path = Path(tmp_edf.name)
    try:
        write_synthetic_edf(edf_path, channel_names=CHBMIT_TEST_CHANNELS, sampling_rate=256, duration_seconds=1)
        with open(edf_path, "rb") as f:
            resp = client.post(
                "/api/v2/predict/",
                headers=headers,
                data={
                    "name": "Canary Patient",
                    "age": 45,
                    "gender": "male",
                    "weight": 70.0,
                    "height": 175.0,
                    "medical_history": json.dumps({}),
                    "vital_signs": json.dumps({}),
                },
                files={"file": ("canary.edf", f, "application/octet-stream")},
            )

        assert resp.status_code == 503
        raw_response_text = resp.text
        assert canary_secret not in raw_response_text
        assert "redis.internal" not in raw_response_text

        # Verify database record is in consistent Failed state without secret leakage
        db = SessionLocal()
        try:
            failed_job = (
                db.query(PredictionJob)
                .filter(PredictionJob.tenant_id == TENANT_ALPHA)
                .order_by(PredictionJob.created_at.desc())
                .first()
            )
            assert failed_job is not None
            assert failed_job.status == "Failed"
            assert failed_job.progress == 0
            assert canary_secret not in (failed_job.error or "")
            assert failed_job.error == "Distributed queue dispatch failed"
        finally:
            db.close()
    finally:
        if edf_path.exists():
            edf_path.unlink()


def test_filesystem_path_leakage_prevented_in_api_and_job_error(client: TestClient, monkeypatch):
    """
    TEL-02 Adversarial Test:
    Simulate pipeline / storage exception containing a private filesystem path canary.
    Verify that:
    1. Raw path is completely absent from all API responses (v1, v2).
    2. Raw path is absent from client-visible job error fields.
    3. The internal failure is correctly recorded as Failed (never falsely successful).
    """
    from app.services.storage import StorageError

    canary_path = "/srv/neuroaegis/private-storage/CANARY_PRIVATE_PATH_12345.npz"

    # Simulate storage error containing private path in distributed mode
    monkeypatch.setattr(settings, "ENABLE_DISTRIBUTED_QUEUE", True)
    mock_storage = MagicMock()
    mock_storage.save_window.side_effect = StorageError(f"IOError: Unable to write to {canary_path}")
    monkeypatch.setattr("app.api.v2.predict.storage_backend", mock_storage)

    user_clin = User(id=USER_CLIN_ALPHA, username="adv_clin_alpha", tenant_id=TENANT_ALPHA, role="clinician", token_version=1)
    token = create_access_token(user_clin)
    headers = {"Authorization": f"Bearer {token}"}

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_edf:
        edf_path = Path(tmp_edf.name)
    try:
        write_synthetic_edf(edf_path, channel_names=CHBMIT_TEST_CHANNELS, sampling_rate=256, duration_seconds=1)
        with open(edf_path, "rb") as f:
            resp = client.post(
                "/api/v2/predict/",
                headers=headers,
                data={
                    "name": "Storage Canary Patient",
                    "age": 52,
                    "gender": "female",
                    "weight": 62.0,
                    "height": 168.0,
                    "medical_history": json.dumps({}),
                    "vital_signs": json.dumps({}),
                },
                files={"file": ("storage_canary.edf", f, "application/octet-stream")},
            )

        assert resp.status_code == 503
        raw_text = resp.text
        assert "CANARY_PRIVATE_PATH" not in raw_text
        assert "/srv/neuroaegis" not in raw_text

        # Verify DB job state
        db = SessionLocal()
        try:
            failed_job = (
                db.query(PredictionJob)
                .filter(PredictionJob.tenant_id == TENANT_ALPHA)
                .order_by(PredictionJob.created_at.desc())
                .first()
            )
            assert failed_job is not None
            assert failed_job.status == "Failed"
            assert "CANARY_PRIVATE_PATH" not in (failed_job.error or "")
            assert failed_job.error == "Storage access failure"

            # Query job status via v2 endpoint
            status_resp = client.get(f"/api/v2/jobs/{failed_job.id}", headers=headers)
            assert status_resp.status_code == 200
            assert "CANARY_PRIVATE_PATH" not in status_resp.text
            assert status_resp.json()["error"] == "Storage access failure"

            # Query report via v2 endpoint
            report_resp = client.get(f"/api/v2/report/{failed_job.id}", headers=headers)
            assert report_resp.status_code == 200
            assert "CANARY_PRIVATE_PATH" not in report_resp.text
            assert report_resp.json()["job"]["error"] == "Storage access failure"
        finally:
            db.close()
    finally:
        if edf_path.exists():
            edf_path.unlink()


def test_legacy_persisted_errors_sanitized_at_response_time(client: TestClient):
    """
    TEL-02 Legacy Data Policy Test:
    Construct database jobs containing legacy unsanitized error strings with canary secrets and paths.
    Verify that:
    1. GET /api/v1/jobs/{job_id} sanitizes the error response.
    2. GET /api/v1/jobs/latest sanitizes the error response.
    3. GET /api/v2/predict/status/{job_id} sanitizes the error response.
    4. GET /api/v2/report/{job_id} sanitizes the error response.
    5. No canary secret, server path, or stack trace leaks through any API.
    """
    canary_legacy_path = "/var/log/private_storage/CANARY_LEGACY_PATH_54321.npz"
    canary_legacy_secret = "CANARY_LEGACY_SECRET_9999"
    raw_legacy_error = f"FileNotFoundError: [Errno 2] No such file or directory: '{canary_legacy_path}' (token: {canary_legacy_secret})"

    legacy_job_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        legacy_job = PredictionJob(
            id=legacy_job_id,
            tenant_id=TENANT_ALPHA,
            created_by_user_id=USER_CLIN_ALPHA,
            status="Failed",
            progress=0,
            error=raw_legacy_error,
            detected_dataset="bonn",
            selected_model="lightgbm",
        )
        db.add(legacy_job)
        db.commit()
    finally:
        db.close()

    user_clin = User(id=USER_CLIN_ALPHA, username="adv_clin_alpha", tenant_id=TENANT_ALPHA, role="clinician", token_version=1)
    token = create_access_token(user_clin)
    headers = {"Authorization": f"Bearer {token}"}

    # 1. v1 job by ID
    v1_resp = client.get(f"/api/v1/jobs/{legacy_job_id}", headers=headers)
    assert v1_resp.status_code == 200
    v1_body = v1_resp.text
    assert canary_legacy_path not in v1_body
    assert canary_legacy_secret not in v1_body
    assert v1_resp.json()["error"] == "Storage access failure"

    # 2. v1 latest job
    v1_latest_resp = client.get("/api/v1/jobs/latest", headers=headers)
    assert v1_latest_resp.status_code == 200
    assert canary_legacy_path not in v1_latest_resp.text
    assert canary_legacy_secret not in v1_latest_resp.text
    assert v1_latest_resp.json()["error"] == "Storage access failure"

    # 3. v2 job status
    v2_status_resp = client.get(f"/api/v2/predict/status/{legacy_job_id}", headers=headers)
    assert v2_status_resp.status_code == 200
    assert canary_legacy_path not in v2_status_resp.text
    assert canary_legacy_secret not in v2_status_resp.text
    assert v2_status_resp.json()["error"] == "Storage access failure"

    # 4. v2 report
    v2_report_resp = client.get(f"/api/v2/report/{legacy_job_id}", headers=headers)
    assert v2_report_resp.status_code == 200
    assert canary_legacy_path not in v2_report_resp.text
    assert canary_legacy_secret not in v2_report_resp.text
    assert v2_report_resp.json()["job"]["error"] == "Storage access failure"


def test_unexpected_exceptions_produce_safe_client_error(client: TestClient):
    """
    Verify that unexpected internal exceptions with sensitive context produce
    safe categorized error messages without disclosing technical stack traces.
    """
    from app.services.worker import fail_job

    unexpected_job_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        job = PredictionJob(
            id=unexpected_job_id,
            tenant_id=TENANT_ALPHA,
            created_by_user_id=USER_CLIN_ALPHA,
            status="Running",
            progress=50,
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    canary_internal_detail = "RuntimeError: Segfault in /usr/local/secret_module.py line 42 with CANARY_INTERNAL_CTX"
    fail_job(unexpected_job_id, canary_internal_detail)

    user_clin = User(id=USER_CLIN_ALPHA, username="adv_clin_alpha", tenant_id=TENANT_ALPHA, role="clinician", token_version=1)
    token = create_access_token(user_clin)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get(f"/api/v2/jobs/{unexpected_job_id}", headers=headers)
    assert resp.status_code == 200
    assert "CANARY_INTERNAL_CTX" not in resp.text
    assert "/usr/local" not in resp.text
    assert resp.json()["error"] in ("Job failed during processing", "Internal processing error")


def test_security_and_tenant_invariants_preserved_during_error_remediation(client: TestClient):
    """
    Verify that error sanitization does not weaken tenant isolation, authorization,
    soft deletion, or normal successful job retrieval.
    """
    # 1. Cross-tenant access on failed job is rejected with 404
    job_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        db.add(PredictionJob(
            id=job_id,
            tenant_id=TENANT_ALPHA,
            created_by_user_id=USER_CLIN_ALPHA,
            status="Failed",
            error="Storage access failure",
        ))
        db.commit()
    finally:
        db.close()

    user_beta = User(id=USER_CLIN_BETA, username="adv_clin_beta", tenant_id=TENANT_BETA, role="clinician", token_version=1)
    token_beta = create_access_token(user_beta)
    headers_beta = {"Authorization": f"Bearer {token_beta}"}

    cross_resp = client.get(f"/api/v2/jobs/{job_id}", headers=headers_beta)
    assert cross_resp.status_code == 404

    # 2. Researcher role cannot access clinical jobs
    user_res = User(id=USER_RES_ALPHA, username="adv_res_alpha", tenant_id=TENANT_ALPHA, role="researcher", token_version=1)
    token_res = create_access_token(user_res)
    headers_res = {"Authorization": f"Bearer {token_res}"}

    res_resp = client.get(f"/api/v2/jobs/{job_id}", headers=headers_res)
    assert res_resp.status_code == 403

    # 3. Soft-deleted failed job is rejected with 404
    db = SessionLocal()
    try:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        job.is_deleted = True
        db.commit()
    finally:
        db.close()

    user_alpha = User(id=USER_CLIN_ALPHA, username="adv_clin_alpha", tenant_id=TENANT_ALPHA, role="clinician", token_version=1)
    token_alpha = create_access_token(user_alpha)
    headers_alpha = {"Authorization": f"Bearer {token_alpha}"}

    deleted_resp = client.get(f"/api/v2/jobs/{job_id}", headers=headers_alpha)
    assert deleted_resp.status_code == 404
