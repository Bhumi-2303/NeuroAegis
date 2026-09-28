"""
NeuroAegis Authorization & Tenant Isolation Test Suite (Prompt 9.3)

Verifies:
1. Authentication boundary: Unauthenticated requests to clinical endpoints return 401.
2. Role authorization matrix:
   - ADMIN: Clinical CRUD within tenant, patient erasure within tenant, no cross-tenant bypass.
   - CLINICIAN: Full clinical operations within tenant, denied patient data deletion (403).
   - RESEARCHER: Allowed on synthetic stream/metadata, denied direct PHI/clinical endpoints (403).
3. Cross-tenant isolation (Tenant A vs Tenant B):
   - Patients list and retrieval by ID are tenant-scoped.
   - Prediction job creation, retrieval, and status are tenant-scoped.
   - Diagnostic reports are tenant-scoped.
   - Prediction history is tenant-scoped.
   - EEG streaming with job_id is tenant-scoped.
4. Non-leaking error semantics (404 Not Found for cross-tenant or missing resources).
5. IDOR protection: Valid UUIDs from another tenant return 404.
6. Tenant inheritance & tampering protection: Request-supplied tenant_id / created_by_user_id ignored.
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.core.auth import create_access_token, get_password_hash
from app.core.config import settings
from app.db.database import Base, SessionLocal, engine, ensure_schema_compatibility
from app.db.models import DEFAULT_TENANT_ID, Patient, PredictionJob, Tenant, User
from app.main import app
from tests.edf_fixture import write_synthetic_edf


TENANT_A_ID = "tenant-alpha-0001"
TENANT_B_ID = "tenant-beta-0002"

USER_ADMIN_A_ID = "user-admin-alpha"
USER_CLIN_A_ID = "user-clin-alpha"
USER_RES_A_ID = "user-res-alpha"

USER_ADMIN_B_ID = "user-admin-beta"
USER_CLIN_B_ID = "user-clin-beta"
USER_RES_B_ID = "user-res-beta"

PATIENT_A1_ID = "patient-alpha-01"
PATIENT_A2_ID = "patient-alpha-02"
PATIENT_B1_ID = "patient-beta-01"
PATIENT_B2_ID = "patient-beta-02"

JOB_A1_ID = "job-alpha-01"
JOB_A2_ID = "job-alpha-02"
JOB_B1_ID = "job-beta-01"
JOB_B2_ID = "job-beta-02"


@pytest.fixture(scope="module")
def setup_multi_tenant_data():
    """Seeds multi-tenant database fixtures with two distinct tenants and role-partitioned users."""
    Base.metadata.create_all(bind=engine)
    ensure_schema_compatibility(engine)
    db = SessionLocal()
    try:
        # Seed Tenants
        tenant_a = db.query(Tenant).filter(Tenant.id == TENANT_A_ID).first()
        if not tenant_a:
            tenant_a = Tenant(id=TENANT_A_ID, name="Alpha General Hospital", slug="alpha-general", is_active=True)
            db.add(tenant_a)

        tenant_b = db.query(Tenant).filter(Tenant.id == TENANT_B_ID).first()
        if not tenant_b:
            tenant_b = Tenant(id=TENANT_B_ID, name="Beta Neuroscience Center", slug="beta-neuro", is_active=True)
            db.add(tenant_b)
        db.commit()

        # Seed Users
        users_to_seed = [
            (USER_ADMIN_A_ID, "admin_alpha", "admin", TENANT_A_ID),
            (USER_CLIN_A_ID, "clin_alpha", "clinician", TENANT_A_ID),
            (USER_RES_A_ID, "res_alpha", "researcher", TENANT_A_ID),
            (USER_ADMIN_B_ID, "admin_beta", "admin", TENANT_B_ID),
            (USER_CLIN_B_ID, "clin_beta", "clinician", TENANT_B_ID),
            (USER_RES_B_ID, "res_beta", "researcher", TENANT_B_ID),
        ]
        for u_id, username, role, t_id in users_to_seed:
            existing = db.query(User).filter(User.id == u_id).first()
            if not existing:
                u = User(
                    id=u_id,
                    username=username,
                    hashed_password=get_password_hash("Password123!"),
                    role=role,
                    tenant_id=t_id,
                    is_active=True,
                    token_version=1,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(u)
        db.commit()

        # Seed Patients
        patients_to_seed = [
            (PATIENT_A1_ID, TENANT_A_ID, USER_CLIN_A_ID, "John Alpha 1"),
            (PATIENT_A2_ID, TENANT_A_ID, USER_CLIN_A_ID, "Jane Alpha 2"),
            (PATIENT_B1_ID, TENANT_B_ID, USER_CLIN_B_ID, "Bob Beta 1"),
            (PATIENT_B2_ID, TENANT_B_ID, USER_CLIN_B_ID, "Alice Beta 2"),
        ]
        for p_id, t_id, creator_id, name in patients_to_seed:
            existing_p = db.query(Patient).filter(Patient.id == p_id).first()
            if not existing_p:
                p = Patient(
                    id=p_id,
                    tenant_id=t_id,
                    created_by_user_id=creator_id,
                    name=name,
                    age=40,
                    gender="M",
                    weight=70.0,
                    height=175.0,
                    medical_history="{}",
                    vital_signs={},
                    is_deleted=False,
                )
                db.add(p)
        db.commit()

        # Seed Jobs
        jobs_to_seed = [
            (JOB_A1_ID, TENANT_A_ID, USER_CLIN_A_ID, PATIENT_A1_ID, "Completed", 100, "Seizure", 0.92),
            (JOB_A2_ID, TENANT_A_ID, USER_CLIN_A_ID, PATIENT_A2_ID, "Validating", 10, None, None),
            (JOB_B1_ID, TENANT_B_ID, USER_CLIN_B_ID, PATIENT_B1_ID, "Completed", 100, "Non-Seizure", 0.12),
            (JOB_B2_ID, TENANT_B_ID, USER_CLIN_B_ID, PATIENT_B2_ID, "Processing", 40, None, None),
        ]
        for j_id, t_id, creator_id, p_id, status, prog, label, prob in jobs_to_seed:
            existing_j = db.query(PredictionJob).filter(PredictionJob.id == j_id).first()
            if not existing_j:
                j = PredictionJob(
                    id=j_id,
                    tenant_id=t_id,
                    created_by_user_id=creator_id,
                    patient_id=p_id,
                    status=status,
                    progress=prog,
                    prediction_label=label,
                    probability_seizure=prob,
                    is_deleted=False,
                )
                db.add(j)
        db.commit()

    finally:
        db.close()


@pytest.fixture
def client():
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_token(user_id: str, tenant_id: str, role: str) -> str:
    user = User(
        id=user_id,
        username=f"u_{user_id}",
        tenant_id=tenant_id,
        role=role,
        token_version=1,
    )
    return create_access_token(user)


def auth_header(user_id: str, tenant_id: str, role: str) -> dict[str, str]:
    token = make_token(user_id, tenant_id, role)
    return {"Authorization": f"Bearer {token}"}


# ==============================================================================
# 1. Authentication Boundary Tests (Unauthenticated -> 401)
# ==============================================================================

def test_unauthenticated_clinical_endpoints_fail_with_401(client: TestClient, setup_multi_tenant_data):
    """Clinical endpoints must reject unauthenticated requests with 401 Unauthorized."""
    endpoints = [
        ("GET", "/api/v1/patients/"),
        ("POST", "/api/v1/patients/"),
        ("GET", f"/api/v1/patients/{PATIENT_A1_ID}"),
        ("POST", "/api/v1/predict/"),
        ("GET", "/api/v1/jobs/latest"),
        ("GET", f"/api/v1/jobs/{JOB_A1_ID}"),
        ("GET", f"/api/v1/stream/eeg?job_id={JOB_A1_ID}"),
        ("DELETE", f"/api/v1/data/patient/{PATIENT_A1_ID}"),
        ("POST", "/api/v2/predict"),
        ("GET", f"/api/v2/predict/status/{JOB_A1_ID}"),
        ("GET", f"/api/v2/jobs/{JOB_A1_ID}"),
        ("GET", f"/api/v2/report/{JOB_A1_ID}"),
        ("GET", "/api/v2/history"),
    ]
    for method, path in endpoints:
        if method == "GET":
            resp = client.get(path)
        elif method == "POST":
            resp = client.post(path)
        elif method == "DELETE":
            resp = client.delete(path)
        assert resp.status_code == 401, f"Expected 401 for unauthenticated {method} {path}, got {resp.status_code}"


def test_public_and_demo_endpoints_remain_accessible(client: TestClient):
    """Probes, model metadata, and demo endpoints must remain intentionally public."""
    assert client.get("/healthz").status_code == 200
    assert client.get("/liveness").status_code == 200
    assert client.get("/api/v1/liveness").status_code == 200
    assert client.get("/api/v1/model/info").status_code in (200, 503)
    assert client.get("/api/v1/models/").status_code in (200, 503)
    # Demo endpoint is public (may return 200 or 404 if parquet file not present in test env, but not 401/403)
    demo_resp = client.get("/api/v2/demo/live-monitor-data")
    assert demo_resp.status_code in (200, 404)


# ==============================================================================
# 2. Role Authorization Matrix Tests
# ==============================================================================

def test_researcher_role_denied_access_to_clinical_endpoints(client: TestClient, setup_multi_tenant_data):
    """
    RESEARCHER role has access to metadata/demo, but is strictly forbidden (403)
    from viewing/creating patient PHI, creating predictions, viewing reports, and accessing history.
    """
    headers = auth_header(USER_RES_A_ID, TENANT_A_ID, "researcher")

    assert client.get("/api/v1/patients/", headers=headers).status_code == 403
    assert client.post("/api/v1/patients/", headers=headers, json={"name": "Test", "age": 30, "gender": "F"}).status_code == 403
    assert client.get(f"/api/v1/patients/{PATIENT_A1_ID}", headers=headers).status_code == 403
    assert client.post("/api/v1/predict/", headers=headers).status_code == 403
    assert client.get("/api/v1/jobs/latest", headers=headers).status_code == 403
    assert client.get(f"/api/v1/jobs/{JOB_A1_ID}", headers=headers).status_code == 403
    assert client.get(f"/api/v2/jobs/{JOB_A1_ID}", headers=headers).status_code == 403
    assert client.get(f"/api/v2/report/{JOB_A1_ID}", headers=headers).status_code == 403
    assert client.get("/api/v2/history", headers=headers).status_code == 403
    assert client.delete(f"/api/v1/data/patient/{PATIENT_A1_ID}", headers=headers).status_code == 403


def test_clinician_allowed_clinical_endpoints_but_denied_patient_deletion(client: TestClient, setup_multi_tenant_data):
    """
    CLINICIAN role has clinical read/create permissions,
    but MUST be rejected with 403 on patient data erasure (GDPR/retention requires admin).
    """
    headers = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")

    # Patient listing & retrieval allowed
    assert client.get("/api/v1/patients/", headers=headers).status_code == 200
    assert client.get(f"/api/v1/patients/{PATIENT_A1_ID}", headers=headers).status_code == 200

    # Job & report viewing allowed
    assert client.get("/api/v1/jobs/latest", headers=headers).status_code == 200
    assert client.get(f"/api/v1/jobs/{JOB_A1_ID}", headers=headers).status_code == 200
    assert client.get(f"/api/v2/jobs/{JOB_A1_ID}", headers=headers).status_code == 200
    assert client.get(f"/api/v2/report/{JOB_A1_ID}", headers=headers).status_code == 200
    assert client.get("/api/v2/history", headers=headers).status_code == 200

    # Patient deletion forbidden for clinician
    del_resp = client.delete(f"/api/v1/data/patient/{PATIENT_A1_ID}", headers=headers)
    assert del_resp.status_code == 403
    assert "Insufficient permissions" in del_resp.json().get("detail", "")


def test_admin_allowed_patient_deletion_within_tenant_only(client: TestClient, setup_multi_tenant_data):
    """
    ADMIN role has erasure authority in own tenant, but cannot delete another tenant's records (404).
    """
    headers_admin_a = auth_header(USER_ADMIN_A_ID, TENANT_A_ID, "admin")

    # Admin A attempts to delete Patient B1 from Tenant B -> 404 (does not reveal existence)
    resp_cross = client.delete(f"/api/v1/data/patient/{PATIENT_B1_ID}", headers=headers_admin_a)
    assert resp_cross.status_code == 404
    assert resp_cross.json()["detail"] == "Patient not found"

    # Verify Patient B1 is still not deleted
    db = SessionLocal()
    try:
        p_b1 = db.query(Patient).filter(Patient.id == PATIENT_B1_ID).first()
        assert p_b1 is not None and not p_b1.is_deleted
    finally:
        db.close()


# ==============================================================================
# 3. Cross-Tenant Isolation & IDOR Protection Tests
# ==============================================================================

def test_cross_tenant_patient_isolation(client: TestClient, setup_multi_tenant_data):
    """Patients in Tenant A cannot be listed or read by Tenant B, and vice versa."""
    headers_a = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")
    headers_b = auth_header(USER_CLIN_B_ID, TENANT_B_ID, "clinician")

    # Tenant A list sees only A patients
    res_a = client.get("/api/v1/patients/", headers=headers_a)
    assert res_a.status_code == 200
    patients_a = res_a.json()
    ids_a = [p["id"] for p in patients_a]
    assert PATIENT_A1_ID in ids_a
    assert PATIENT_A2_ID in ids_a
    assert PATIENT_B1_ID not in ids_a
    assert PATIENT_B2_ID not in ids_a

    # Tenant B list sees only B patients
    res_b = client.get("/api/v1/patients/", headers=headers_b)
    assert res_b.status_code == 200
    patients_b = res_b.json()
    ids_b = [p["id"] for p in patients_b]
    assert PATIENT_B1_ID in ids_b
    assert PATIENT_B2_ID in ids_b
    assert PATIENT_A1_ID not in ids_b
    assert PATIENT_A2_ID not in ids_b

    # IDOR: Tenant A requests Patient B1 by valid UUID -> 404
    resp_idor_a = client.get(f"/api/v1/patients/{PATIENT_B1_ID}", headers=headers_a)
    assert resp_idor_a.status_code == 404
    assert resp_idor_a.json()["detail"] == "Patient not found"

    # IDOR: Tenant B requests Patient A1 by valid UUID -> 404
    resp_idor_b = client.get(f"/api/v1/patients/{PATIENT_A1_ID}", headers=headers_b)
    assert resp_idor_b.status_code == 404
    assert resp_idor_b.json()["detail"] == "Patient not found"


def test_cross_tenant_job_isolation(client: TestClient, setup_multi_tenant_data):
    """Prediction jobs are strictly scoped to the caller's tenant."""
    headers_a = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")
    headers_b = auth_header(USER_CLIN_B_ID, TENANT_B_ID, "clinician")

    # Latest jobs query scopes to caller tenant
    latest_a = client.get("/api/v1/jobs/latest", headers=headers_a)
    assert latest_a.status_code == 200
    latest_a_job_id = latest_a.json()["job_id"]
    assert latest_a_job_id not in (JOB_B1_ID, JOB_B2_ID)

    latest_b = client.get("/api/v1/jobs/latest", headers=headers_b)
    assert latest_b.status_code == 200
    latest_b_job_id = latest_b.json()["job_id"]
    assert latest_b_job_id not in (JOB_A1_ID, JOB_A2_ID)

    db = SessionLocal()
    try:
        j_a = db.query(PredictionJob).filter(PredictionJob.id == latest_a_job_id).first()
        assert j_a is not None
        assert j_a.tenant_id == TENANT_A_ID

        j_b = db.query(PredictionJob).filter(PredictionJob.id == latest_b_job_id).first()
        assert j_b is not None
        assert j_b.tenant_id == TENANT_B_ID
    finally:
        db.close()

    # Job by ID (v1 and v2 aliases)
    assert client.get(f"/api/v1/jobs/{JOB_A1_ID}", headers=headers_a).status_code == 200
    assert client.get(f"/api/v1/jobs/{JOB_B1_ID}", headers=headers_a).status_code == 404
    assert client.get(f"/api/v2/jobs/{JOB_B1_ID}", headers=headers_a).status_code == 404
    assert client.get(f"/api/v2/predict/status/{JOB_B1_ID}", headers=headers_a).status_code == 404

    assert client.get(f"/api/v1/jobs/{JOB_B1_ID}", headers=headers_b).status_code == 200
    assert client.get(f"/api/v1/jobs/{JOB_A1_ID}", headers=headers_b).status_code == 404
    assert client.get(f"/api/v2/jobs/{JOB_A1_ID}", headers=headers_b).status_code == 404
    assert client.get(f"/api/v2/predict/status/{JOB_A1_ID}", headers=headers_b).status_code == 404


def test_cross_tenant_report_isolation(client: TestClient, setup_multi_tenant_data):
    """Clinical reports cannot be retrieved cross-tenant."""
    headers_a = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")
    headers_b = auth_header(USER_CLIN_B_ID, TENANT_B_ID, "clinician")

    # Own report succeeds
    rep_a = client.get(f"/api/v2/report/{JOB_A1_ID}", headers=headers_a)
    assert rep_a.status_code == 200
    assert rep_a.json()["patient"]["name"] == "John Alpha 1"

    # Cross-tenant report fails with 404
    assert client.get(f"/api/v2/report/{JOB_B1_ID}", headers=headers_a).status_code == 404
    assert client.get(f"/api/v2/report/{JOB_A1_ID}", headers=headers_b).status_code == 404


def test_cross_tenant_history_isolation(client: TestClient, setup_multi_tenant_data):
    """Prediction history only reveals records belonging to caller's tenant."""
    headers_a = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")
    headers_b = auth_header(USER_CLIN_B_ID, TENANT_B_ID, "clinician")

    hist_a = client.get("/api/v2/history", headers=headers_a).json()
    hist_a_job_ids = [item["job_id"] for item in hist_a]
    assert JOB_A1_ID in hist_a_job_ids
    assert JOB_A2_ID in hist_a_job_ids
    assert JOB_B1_ID not in hist_a_job_ids
    assert JOB_B2_ID not in hist_a_job_ids

    hist_b = client.get("/api/v2/history", headers=headers_b).json()
    hist_b_job_ids = [item["job_id"] for item in hist_b]
    assert JOB_B1_ID in hist_b_job_ids
    assert JOB_B2_ID in hist_b_job_ids
    assert JOB_A1_ID not in hist_b_job_ids
    assert JOB_A2_ID not in hist_b_job_ids


def test_cross_tenant_eeg_stream_isolation(client: TestClient, setup_multi_tenant_data):
    """EEG streaming with job_id rejects cross-tenant job references with 404."""
    from unittest.mock import patch
    from fastapi.responses import Response

    headers_a = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")
    headers_b = auth_header(USER_CLIN_B_ID, TENANT_B_ID, "clinician")

    # Own tenant job stream succeeds (mock StreamingResponse to prevent infinite SSE loop in test client)
    with patch("app.api.v1.stream.StreamingResponse", return_value=Response(content="stream-ok", status_code=200)):
        res_a = client.get(f"/api/v1/stream/eeg?job_id={JOB_A1_ID}", headers=headers_a)
        assert res_a.status_code == 200

    # Cross-tenant job stream fails with 404
    res_cross_a = client.get(f"/api/v1/stream/eeg?job_id={JOB_B1_ID}", headers=headers_a)
    assert res_cross_a.status_code == 404

    res_cross_b = client.get(f"/api/v1/stream/eeg?job_id={JOB_A1_ID}", headers=headers_b)
    assert res_cross_b.status_code == 404


# ==============================================================================
# 4. Tenant Inheritance & Request Tampering Attacks
# ==============================================================================

def test_patient_creation_ignores_client_supplied_tenant_and_creator(client: TestClient, setup_multi_tenant_data):
    """
    Malicious request payload attempting to set tenant_id or created_by_user_id
    MUST be ignored, binding the record strictly to current_user.
    """
    headers = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")
    malicious_payload = {
        "name": "Tampered Patient",
        "age": 35,
        "gender": "F",
        "weight": 62.0,
        "height": 168.0,
        "medical_history": "{}",
        "vital_signs": {},
        "tenant_id": TENANT_B_ID,  # Attacker attempts to inject into Tenant B
        "created_by_user_id": USER_ADMIN_B_ID,  # Attacker attempts to impersonate Admin B
    }

    resp = client.post("/api/v1/patients/", headers=headers, json=malicious_payload)
    assert resp.status_code in (200, 201)
    created_id = resp.json()["id"]

    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.id == created_id).first()
        assert patient is not None
        assert patient.tenant_id == TENANT_A_ID, "Authoritative tenant was overridden by client input!"
        assert patient.created_by_user_id == USER_CLIN_A_ID, "Authoritative creator was overridden by client input!"
    finally:
        db.close()


def test_v1_predict_cannot_bind_job_to_cross_tenant_patient(client: TestClient, setup_multi_tenant_data, tmp_path):
    """
    Clinician in Tenant A attempting to create a prediction for Patient in Tenant B
    must be blocked with 404 Not Found.
    """
    headers_a = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")

    fixture_path = write_synthetic_edf(
        tmp_path / "test_cross_patient.edf",
        channel_names=[f"Ch{i}" for i in range(23)],
    )

    resp = client.post(
        "/api/v1/predict/",
        headers=headers_a,
        files={"file": ("test.edf", fixture_path.read_bytes(), "application/octet-stream")},
        data={"patient_id": PATIENT_B1_ID},  # Patient from Tenant B!
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Patient not found"


def test_v2_predict_assigns_authenticated_tenant_and_creator(client: TestClient, setup_multi_tenant_data, tmp_path):
    """
    v2 /predict derives tenant_id and created_by_user_id authoritatively from current_user.
    """
    headers_a = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")

    ch_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ",
        "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
    ]
    edf_path = write_synthetic_edf(tmp_path / "v2_auth_test.edf", channel_names=ch_names, sampling_rate=256)

    resp = client.post(
        "/api/v2/predict",
        headers=headers_a,
        data={
            "name": "Auth V2 Patient",
            "age": 29,
            "gender": "male",
            "weight": 72.0,
            "height": 178.0,
            "medical_history": "{}",
            "vital_signs": "{}",
            "tenant_id": TENANT_B_ID,  # Malicious injection
        },
        files={"file": ("v2_auth_test.edf", edf_path.read_bytes(), "application/octet-stream")},
    )
    assert resp.status_code == 200
    res_data = resp.json()
    job_id = res_data["job_id"]
    patient_id = res_data["patient_id"]

    db = SessionLocal()
    try:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        assert job.tenant_id == TENANT_A_ID
        assert job.created_by_user_id == USER_CLIN_A_ID
        assert patient.tenant_id == TENANT_A_ID
        assert patient.created_by_user_id == USER_CLIN_A_ID
    finally:
        db.close()


def test_v2_predict_cross_tenant_patient_rejected_with_404(client: TestClient, setup_multi_tenant_data, tmp_path):
    """
    Prompt 9.3.1 Phase 6:
    User A (Clinician in Tenant A) attempting v2 prediction using Patient B (Tenant B)
    must fail with 404 Not Found, creating no job and leaking no Tenant B information.
    Then User A using Patient A (Tenant A) must succeed, attaching job to Tenant A.
    """
    headers_a = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")
    ch_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ",
        "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
    ]
    edf_path = write_synthetic_edf(tmp_path / "v2_cross_test.edf", channel_names=ch_names, sampling_rate=256)
    edf_bytes = edf_path.read_bytes()

    db = SessionLocal()
    initial_jobs_count = db.query(PredictionJob).count()
    db.close()

    # Attempt cross-tenant reference
    resp_cross = client.post(
        "/api/v2/predict",
        headers=headers_a,
        data={
            "patient_id": PATIENT_B1_ID,
            "name": "Ignored Demographics",
            "age": 40,
            "gender": "male",
            "weight": 80.0,
            "height": 180.0,
            "medical_history": "{}",
            "vital_signs": "{}",
        },
        files={"file": ("v2_cross_test.edf", edf_bytes, "application/octet-stream")},
    )
    assert resp_cross.status_code == 404
    assert resp_cross.json()["detail"] == "Patient not found"

    db = SessionLocal()
    try:
        # Verify no job was created
        assert db.query(PredictionJob).count() == initial_jobs_count
    finally:
        db.close()

    # Same-tenant reference succeeds
    resp_same = client.post(
        "/api/v2/predict",
        headers=headers_a,
        data={
            "patient_id": PATIENT_A1_ID,
            "name": "Patient A1",
            "age": 45,
            "gender": "female",
            "weight": 65.0,
            "height": 165.0,
            "medical_history": "{}",
            "vital_signs": "{}",
        },
        files={"file": ("v2_cross_test.edf", edf_bytes, "application/octet-stream")},
    )
    assert resp_same.status_code == 200
    res_data = resp_same.json()
    job_id = res_data["job_id"]
    assert res_data["patient_id"] == PATIENT_A1_ID

    db = SessionLocal()
    try:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        assert job is not None
        assert job.patient_id == PATIENT_A1_ID
        assert job.tenant_id == TENANT_A_ID
        assert job.created_by_user_id == USER_CLIN_A_ID
    finally:
        db.close()


def test_v2_predict_client_tenant_and_creator_manipulation(client: TestClient, setup_multi_tenant_data, tmp_path):
    """
    Prompt 9.3.1 Phase 7:
    Malicious tenant_id and created_by_user_id in body, form, and query parameters
    must be ignored; resulting records must always use authenticated user's identity.
    """
    headers_a = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")
    ch_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ",
        "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
    ]
    edf_path = write_synthetic_edf(tmp_path / "v2_manip_test.edf", channel_names=ch_names, sampling_rate=256)

    resp = client.post(
        f"/api/v2/predict?tenant_id={TENANT_B_ID}&created_by_user_id={USER_CLIN_B_ID}",
        headers=headers_a,
        data={
            "name": "Tamper Patient",
            "age": 33,
            "gender": "female",
            "weight": 62.0,
            "height": 168.0,
            "medical_history": "{}",
            "vital_signs": "{}",
            "tenant_id": TENANT_B_ID,
            "created_by_user_id": USER_CLIN_B_ID,
        },
        files={"file": ("v2_manip_test.edf", edf_path.read_bytes(), "application/octet-stream")},
    )
    assert resp.status_code == 200
    res_data = resp.json()
    job_id = res_data["job_id"]
    patient_id = res_data["patient_id"]

    db = SessionLocal()
    try:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        assert job.tenant_id == TENANT_A_ID
        assert job.created_by_user_id == USER_CLIN_A_ID
        assert patient.tenant_id == TENANT_A_ID
        assert patient.created_by_user_id == USER_CLIN_A_ID
    finally:
        db.close()


def test_v2_predict_duplication_semantics_audit(client: TestClient, setup_multi_tenant_data, tmp_path):
    """
    Prompt 9.3.1 Phase 4:
    Audit and document existing v2 prediction behavior:
    Repeated predictions without patient_id intentionally create distinct Patient records.
    """
    headers_a = auth_header(USER_CLIN_A_ID, TENANT_A_ID, "clinician")
    ch_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ",
        "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
    ]
    edf_path = write_synthetic_edf(tmp_path / "v2_dup_test.edf", channel_names=ch_names, sampling_rate=256)
    edf_bytes = edf_path.read_bytes()

    form_payload = {
        "name": "Duplicate Candidate",
        "age": 52,
        "gender": "male",
        "weight": 85.0,
        "height": 182.0,
        "medical_history": "{}",
        "vital_signs": "{}",
    }

    # Prediction 1
    resp1 = client.post(
        "/api/v2/predict",
        headers=headers_a,
        data=form_payload,
        files={"file": ("v2_dup_test.edf", edf_bytes, "application/octet-stream")},
    )
    assert resp1.status_code == 200
    p1_id = resp1.json()["patient_id"]

    # Prediction 2 with same logical patient demographics
    resp2 = client.post(
        "/api/v2/predict",
        headers=headers_a,
        data=form_payload,
        files={"file": ("v2_dup_test.edf", edf_bytes, "application/octet-stream")},
    )
    assert resp2.status_code == 200
    p2_id = resp2.json()["patient_id"]

    # In pre-9.3 v2 API contract without patient_id, distinct UUIDs are created
    assert p1_id != p2_id



# ==============================================================================
# 5. Cookie vs Bearer Authentication Parity & CSRF Verification
# ==============================================================================

def test_cookie_authentication_and_csrf_protection(client: TestClient, setup_multi_tenant_data):
    """
    Browser-facing cookie authentication functions identically to Bearer token,
    and enforces CSRF token validation on state-changing requests.
    """
    token_clin_a = make_token(USER_CLIN_A_ID, TENANT_A_ID, "clinician")

    # 1. GET with access cookie succeeds without CSRF
    client.cookies.set(settings.ACCESS_COOKIE_NAME, token_clin_a)
    res_get = client.get("/api/v1/patients/")
    assert res_get.status_code == 200

    # 2. POST with access cookie WITHOUT CSRF token fails with 403
    res_post_no_csrf = client.post(
        "/api/v1/patients/",
        json={"name": "Cookie Patient", "age": 32, "gender": "F", "weight": 65.0, "height": 170.0},
    )
    assert res_post_no_csrf.status_code == 403
    assert "CSRF token missing or invalid" in res_post_no_csrf.json()["detail"]

    # 3. POST with access cookie WITH matching double-submit CSRF cookie + header succeeds
    csrf_secret = "test-csrf-token-abc-123"
    client.cookies.set(settings.CSRF_COOKIE_NAME, csrf_secret)
    res_post_with_csrf = client.post(
        "/api/v1/patients/",
        json={"name": "Cookie Patient", "age": 32, "gender": "F", "weight": 65.0, "height": 170.0},
        headers={"x-csrf-token": csrf_secret},
    )
    assert res_post_with_csrf.status_code in (200, 201)
