from __future__ import annotations
import asyncio
import io
import json
import logging
import re
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.auth import create_access_token
from app.core.config import settings
from app.core.logging import (
    StructuredJsonFormatter,
    get_job_id,
    get_request_id,
    get_tenant_id,
    get_worker_id,
    reset_logging_context,
    set_logging_context,
    setup_logging,
    validate_or_generate_request_id,
)
from app.db.database import Base, SessionLocal, engine, ensure_schema_compatibility
from app.db.models import Patient, PredictionJob, Tenant, User
from app.main import app
from app.services.job_service import run_prediction_pipeline
from app.services.worker import run_prediction_task
from tests.edf_fixture import write_synthetic_edf

UUID4_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE)

CORR_TENANT = "tenant_corr_01"
CORR_USER = "user_corr_clinician"


@pytest.fixture(scope="module")
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    ensure_schema_compatibility(engine)
    db = SessionLocal()
    try:
        if not db.query(Tenant).filter(Tenant.id == CORR_TENANT).first():
            db.add(Tenant(id=CORR_TENANT, name="Correlation Clinic", slug=CORR_TENANT, is_active=True))
        if not db.query(User).filter(User.id == CORR_USER).first():
            db.add(
                User(
                    id=CORR_USER,
                    tenant_id=CORR_TENANT,
                    username="corr_clinician",
                    hashed_password="hashed_pass_test",
                    role="clinician",
                    is_active=True,
                )
            )
        db.commit()
    finally:
        db.close()


@pytest.fixture
def auth_headers(setup_test_db):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == CORR_USER).first()
        token = create_access_token(user)
    finally:
        db.close()
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    return TestClient(app)


# ==============================================================================
# 1. Request ID Validation & Normalization Unit Tests
# ==============================================================================

def test_validate_or_generate_request_id_empty():
    req_id = validate_or_generate_request_id(None)
    assert req_id is not None
    assert UUID4_PATTERN.match(req_id)

    req_id_empty = validate_or_generate_request_id("")
    assert req_id_empty is not None
    assert UUID4_PATTERN.match(req_id_empty)


def test_validate_or_generate_request_id_valid():
    valid_ids = [
        "req-12345",
        "custom_correlation_id_001",
        "abc-DEF-123_456",
        str(uuid.uuid4()),
        "x" * 64,  # Exactly 64 chars
    ]
    for test_id in valid_ids:
        assert validate_or_generate_request_id(test_id) == test_id


def test_validate_or_generate_request_id_invalid_characters():
    invalid_ids = [
        "req with spaces",
        "<script>alert(1)</script>",
        "req;drop table",
        "req/id/slash",
        "req@symbol",
        "req\nid\r",
        "ü_unicode_id",
    ]
    for test_id in invalid_ids:
        sanitized = validate_or_generate_request_id(test_id)
        assert sanitized != test_id
        assert UUID4_PATTERN.match(sanitized)


def test_validate_or_generate_request_id_oversized():
    oversized = "a" * 65
    sanitized = validate_or_generate_request_id(oversized)
    assert sanitized != oversized
    assert UUID4_PATTERN.match(sanitized)


# ==============================================================================
# 2. Middleware Request Correlation Tests
# ==============================================================================

def test_middleware_generates_uuid_when_header_missing(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-request-id" in response.headers
    req_id = response.headers["x-request-id"]
    assert UUID4_PATTERN.match(req_id)


def test_middleware_preserves_valid_header(client):
    test_id = "test-req-trace-42"
    response = client.get("/health", headers={"X-Request-ID": test_id})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == test_id


def test_middleware_replaces_invalid_header(client):
    test_id = "bad id with spaces and <chars>"
    response = client.get("/health", headers={"X-Request-ID": test_id})
    assert response.status_code == 200
    returned_id = response.headers.get("x-request-id")
    assert returned_id != test_id
    assert UUID4_PATTERN.match(returned_id)


def test_middleware_replaces_oversized_header(client):
    oversized_id = "k" * 128
    response = client.get("/health", headers={"X-Request-ID": oversized_id})
    assert response.status_code == 200
    returned_id = response.headers.get("x-request-id")
    assert returned_id != oversized_id
    assert UUID4_PATTERN.match(returned_id)


def test_middleware_attaches_request_id_to_404(client):
    response = client.get("/non-existent-endpoint-path-999")
    assert response.status_code == 404
    assert "x-request-id" in response.headers
    assert UUID4_PATTERN.match(response.headers["x-request-id"])


def test_middleware_attaches_request_id_to_401(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401
    assert "x-request-id" in response.headers


def test_middleware_attaches_request_id_to_400(client, auth_headers):
    # Missing required fields
    response = client.post("/api/v1/predict/", data={}, headers=auth_headers)
    assert response.status_code == 422 or response.status_code == 400
    assert "x-request-id" in response.headers


def test_middleware_attaches_request_id_on_unhandled_exception(client):
    # Route that triggers an unhandled exception
    @app.get("/test-unhandled-exception-probe")
    def _faulty_endpoint():
        raise RuntimeError("Simulated internal error")

    response = client.get("/test-unhandled-exception-probe", headers={"X-Request-ID": "err-trace-500"})
    assert response.status_code == 500
    assert response.headers.get("x-request-id") == "err-trace-500"


def test_middleware_context_cleans_up_in_finally(client):
    assert get_request_id() is None
    response = client.get("/health")
    assert response.status_code == 200
    # Outside the request handler lifecycle, ContextVar must be reset
    assert get_request_id() is None


def test_cors_middleware_exposes_request_id(client):
    response = client.get(
        "/healthz",
        headers={"Origin": "http://localhost:3000"},
    )
    exposed = response.headers.get("access-control-expose-headers", "")
    assert "X-Request-ID" in exposed or "x-request-id" in exposed.lower()


# ==============================================================================
# 3. Structured JSON Logging Tests
# ==============================================================================

def test_structured_json_formatter_all_fields():
    formatter = StructuredJsonFormatter()
    tokens = set_logging_context(
        request_id="req-test-999",
        job_id="job-test-888",
        tenant_id="tenant-test-777",
        worker_id="worker-test-666",
    )
    try:
        record = logging.LogRecord(
            name="neuroaegis.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=100,
            msg="Processing clinical event for job %s",
            args=("job-test-888",),
            exc_info=None,
        )
        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["level"] == "INFO"
        assert data["logger"] == "neuroaegis.test"
        assert data["message"] == "Processing clinical event for job job-test-888"
        assert data["request_id"] == "req-test-999"
        assert data["job_id"] == "job-test-888"
        assert data["tenant_id"] == "tenant-test-777"
        assert data["worker_id"] == "worker-test-666"
        assert "timestamp" in data
        assert "exception_type" not in data
    finally:
        reset_logging_context(tokens)


def test_structured_json_formatter_unbound_context():
    formatter = StructuredJsonFormatter()
    # Ensure empty context
    assert get_request_id() is None
    record = logging.LogRecord(
        name="neuroaegis.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=101,
        msg="Unbound context warning",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["level"] == "WARNING"
    assert data["message"] == "Unbound context warning"
    assert data["request_id"] is None
    assert data["job_id"] is None
    assert data["tenant_id"] is None
    assert data["worker_id"] is None


def test_structured_json_formatter_exception_sanitization():
    formatter = StructuredJsonFormatter()
    try:
        raise ValueError("Invalid EDF header /Volumes/SECRET/data/eeg.edf")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="neuroaegis.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=102,
        msg="Pipeline encountered error",
        args=(),
        exc_info=exc_info,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["level"] == "ERROR"
    assert data["exception_type"] == "ValueError"
    assert data["exception_category"] == "Signal processing failed"
    # Exception message is present in data and paths are scrubbed
    assert "exception" in data
    assert "/Volumes/SECRET" not in data["exception"]


def test_structured_json_formatter_scrubs_secrets():
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="neuroaegis.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=103,
        msg="Connection broker: redis://default:supersecretpassword@redis:6379/0 and token Bearer eyJhbGciOiJIUzI1NiJ9.secretpayload.sig",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert "supersecretpassword" not in data["message"]
    assert "eyJhbGciOiJIUzI1NiJ9.secretpayload.sig" not in data["message"]
    assert "redis://default:***@redis:6379/0" in data["message"]
    assert "[MASKED_TOKEN]" in data["message"]


def test_setup_logging_idempotence():
    # Calling setup_logging multiple times should not add duplicate handlers
    root_logger = logging.getLogger()
    initial_handlers_count = len(root_logger.handlers)

    setup_logging(force=False)
    setup_logging(force=False)
    setup_logging(force=False)

    assert len(root_logger.handlers) == initial_handlers_count


# ==============================================================================
# 4. Pipeline & Worker Correlation Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_in_process_prediction_pipeline_sets_and_resets_context(setup_test_db):
    db = SessionLocal()
    job_id = f"job-pipeline-test-{uuid.uuid4().hex[:6]}"
    req_id = f"req-pipeline-test-{uuid.uuid4().hex[:6]}"

    job = PredictionJob(
        id=job_id,
        tenant_id=CORR_TENANT,
        created_by_user_id=CORR_USER,
        status="Validating",
        progress=0,
        request_id=req_id,
    )
    db.add(job)
    db.commit()
    db.close()

    context_during_pipeline = {}

    def mock_predict(*args, **kwargs):
        # Capture context during pipeline inference
        context_during_pipeline["request_id"] = get_request_id()
        context_during_pipeline["job_id"] = get_job_id()
        return {
            "prediction": {"label": "non_seizure", "probabilities": {"seizure": 0.05}},
            "confidence": {"band": "high"},
            "explanation": {"baseValue": 0.5, "features": []},
        }

    synthetic_eeg = np.random.randn(16, 256 * 3)
    channel_names = [f"Ch{i}" for i in range(16)]

    with patch("app.services.prediction.prediction_router.prediction_router.get_predictor") as mock_router:
        mock_predictor = MagicMock()
        mock_predictor.get_prediction = mock_predict
        mock_router.return_value = mock_predictor

        await run_prediction_pipeline(
            job_id=job_id,
            eeg_data=synthetic_eeg,
            channel_names=channel_names,
            fs=256.0,
            dataset_name="bonn",
            request_id=req_id,
        )

    # In-pipeline context captured
    assert context_during_pipeline.get("request_id") == req_id
    assert context_during_pipeline.get("job_id") == job_id

    # Post-pipeline context must be cleanly reset
    assert get_request_id() is None
    assert get_job_id() is None


@pytest.mark.asyncio
async def test_worker_task_sets_and_resets_logging_context(setup_test_db):
    db = SessionLocal()
    job_id = f"job-worker-test-{uuid.uuid4().hex[:6]}"
    req_id = f"req-worker-test-{uuid.uuid4().hex[:6]}"

    job = PredictionJob(
        id=job_id,
        tenant_id=CORR_TENANT,
        created_by_user_id=CORR_USER,
        status="Validating",
        progress=0,
        request_id=req_id,
    )
    db.add(job)
    db.commit()
    db.close()

    context_during_task = {}

    async def mock_run_pipeline(*args, **kwargs):
        context_during_task["worker_id"] = get_worker_id()
        context_during_task["job_id"] = get_job_id()
        context_during_task["request_id"] = get_request_id()

    with patch("app.services.worker.load_staged_payload") as mock_load, \
         patch("app.services.worker.run_prediction_pipeline", side_effect=mock_run_pipeline), \
         patch("app.services.storage.storage_backend.delete_window"):

        mock_load.return_value = (
            np.zeros((16, 256 * 3)),
            [f"Ch{i}" for i in range(16)],
            256.0,
            {},
        )

        ctx = {}
        result = await run_prediction_task(
            ctx=ctx,
            job_id=job_id,
            staged_path="staged/fake.npz",
            dataset_name="bonn",
            request_id=req_id,
        )

    assert result["status"] == "completed"
    assert context_during_task.get("request_id") == req_id
    assert context_during_task.get("job_id") == job_id
    assert context_during_task.get("worker_id") is not None

    # Post-task context must be cleanly reset
    assert get_request_id() is None
    assert get_job_id() is None
    assert get_worker_id() is None


# ==============================================================================
# 5. API v2 Prediction Job Request ID Persistence Test
# ==============================================================================

def test_api_v2_predict_persists_request_id(client, auth_headers, tmp_path):
    from app.services.prediction.prediction_router import prediction_router
    if not prediction_router.is_loaded:
        prediction_router.load_all_models()

    edf_path = tmp_path / "test_corr.edf"
    ch_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ",
        "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
    ]
    write_synthetic_edf(
        path=edf_path,
        channel_names=ch_names,
        duration_seconds=1,
        sampling_rate=256,
    )

    custom_request_id = "req-audit-v2-persistence"

    with open(edf_path, "rb") as f:
        files = {"file": ("test_corr.edf", f, "application/octet-stream")}
        data = {
            "name": "Correlation Patient",
            "age": "45",
            "gender": "M",
            "weight": "70",
            "height": "175",
            "medical_history": "{}",
            "vital_signs": "{}",
        }
        headers = {**auth_headers, "X-Request-ID": custom_request_id}
        response = client.post("/api/v2/predict", data=data, files=files, headers=headers)

    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_request_id
    job_id = response.json()["job_id"]

    # Verify database persistence of request_id
    db = SessionLocal()
    try:
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        assert job is not None
        assert job.request_id == custom_request_id
    finally:
        db.close()

    # Verify status endpoint returns request_id
    status_res = client.get(f"/api/v2/jobs/{job_id}", headers=auth_headers)
    assert status_res.status_code == 200
    assert status_res.json().get("request_id") == custom_request_id
