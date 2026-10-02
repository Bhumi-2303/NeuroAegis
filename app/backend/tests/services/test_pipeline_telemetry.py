from __future__ import annotations
import asyncio
import datetime
import io
import time
import uuid
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.auth import create_access_token
from app.core.config import settings
from app.core.errors import (
    SAFE_ERROR_INFERENCE,
    SAFE_ERROR_INTERNAL,
    SAFE_ERROR_LEASE_EXPIRED,
    SAFE_ERROR_QUEUE_DISPATCH,
)
from app.core.telemetry import (
    JOB_EXECUTION_SECONDS,
    JOBS_COMPLETED_TOTAL,
    JOBS_CREATED_TOTAL,
    JOBS_FAILED_TOTAL,
    JOBS_IN_PROGRESS,
    JOBS_REAPED_TOTAL,
    PIPELINE_STAGE_SECONDS,
    QUEUE_DISPATCH_FAILURES_TOTAL,
    QUEUE_WAIT_SECONDS,
    WORKER_CLAIM_FAILURES_TOTAL,
    get_telemetry_registry,
)
from app.db.database import Base, ensure_schema_compatibility
from app.db.models import Patient, PredictionJob, Tenant, User
from app.main import app
from app.services.job_recovery import (
    reap_stale_jobs,
    recover_job_as_failed,
)
from app.services.job_service import run_prediction_pipeline
from app.services.prediction.prediction_router import prediction_router
from app.services.storage import LocalStorageBackend
from app.services.worker import run_prediction_task
from tests.edf_fixture import write_synthetic_edf

TEST_TENANT_ID = "00000000-0000-0000-0000-000000000099"
TEST_USER_ID = "user_telemetry_test_01"


@pytest.fixture(autouse=True)
def reset_telemetry():
    """Ensure clean metrics for each test and loaded models."""
    get_telemetry_registry().clear()
    if not prediction_router.is_loaded:
        prediction_router.load_all_models()
    yield
    get_telemetry_registry().clear()


@pytest.fixture
def isolated_db(monkeypatch):
    """Provides an isolated SQLite DB with StaticPool."""
    from sqlalchemy.pool import StaticPool
    from app.db.database import get_db

    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    ensure_schema_compatibility(test_engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr("app.services.worker.SessionLocal", TestingSession)
    monkeypatch.setattr("app.services.job_service.SessionLocal", TestingSession)
    monkeypatch.setattr("app.services.job_recovery.SessionLocal", TestingSession)

    def _get_test_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_test_db

    session = TestingSession()
    # Seed tenant & user
    tenant = Tenant(id=TEST_TENANT_ID, name="Telemetry Clinic", slug="telemetry-clinic", is_active=True)
    user = User(
        id=TEST_USER_ID,
        tenant_id=TEST_TENANT_ID,
        username="telem_user",
        hashed_password="pw",
        role="clinician",
        is_active=True,
    )
    session.add(tenant)
    session.add(user)
    session.commit()

    try:
        yield session
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def test_client(isolated_db):
    return TestClient(app)


@pytest.fixture
def auth_headers(isolated_db):
    from app.core.auth import get_current_user
    user = isolated_db.query(User).filter(User.id == TEST_USER_ID).first()
    app.dependency_overrides[get_current_user] = lambda: user
    token = create_access_token(user)
    yield {"Authorization": f"Bearer {token}"}
    app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.asyncio
async def test_in_process_pipeline_telemetry(isolated_db):
    """Test full in-process pipeline execution latency and stage tracking."""
    job_id = f"job_telem_{uuid.uuid4().hex[:8]}"
    job = PredictionJob(
        id=job_id,
        tenant_id=TEST_TENANT_ID,
        status="Pending",
        detected_dataset="bonn",
    )
    isolated_db.add(job)
    isolated_db.commit()

    eeg_data = np.random.randn(1, 4097).astype(np.float32)
    channel_names = ["EEG1"]
    fs = 173.61

    # Before running, in_progress gauge is 0
    assert JOBS_IN_PROGRESS.get_value(labels={"execution_mode": "in_process"}) == 0.0

    await run_prediction_pipeline(
        job_id=job_id,
        eeg_data=eeg_data,
        channel_names=channel_names,
        fs=fs,
        dataset_name="bonn",
    )

    # In-progress gauge decremented back to 0
    assert JOBS_IN_PROGRESS.get_value(labels={"execution_mode": "in_process"}) == 0.0

    # Total completed incremented
    assert JOBS_COMPLETED_TOTAL.get_value(labels={"dataset": "bonn", "execution_mode": "in_process"}) == 1.0

    # Execution duration recorded
    exec_stats = JOB_EXECUTION_SECONDS.get_stats(
        labels={"dataset": "bonn", "execution_mode": "in_process", "status": "success"}
    )
    assert exec_stats["count"] == 1
    assert exec_stats["sum"] > 0.0

    # Pipeline stages recorded
    for stage in ("preprocessing", "feature_extraction", "model_inference", "explanation", "persistence"):
        stage_stats = PIPELINE_STAGE_SECONDS.get_stats(
            labels={"stage": stage, "dataset": "bonn", "status": "success"}
        )
        assert stage_stats["count"] == 1, f"Expected stage {stage} to have 1 observation"
        assert stage_stats["sum"] >= 0.0


@pytest.mark.asyncio
async def test_pipeline_failure_telemetry(isolated_db):
    """Test pipeline failure records failed latency and increments JOBS_FAILED_TOTAL."""
    job_id = f"job_fail_{uuid.uuid4().hex[:8]}"
    job = PredictionJob(
        id=job_id,
        tenant_id=TEST_TENANT_ID,
        status="Pending",
        detected_dataset="bonn",
    )
    isolated_db.add(job)
    isolated_db.commit()

    eeg_data = np.random.randn(1, 4097).astype(np.float32)

    class ModelInferenceError(Exception):
        pass

    with patch("app.services.job_service._run_inference", side_effect=ModelInferenceError("Model inference failed")):
        await run_prediction_pipeline(
            job_id=job_id,
            eeg_data=eeg_data,
            channel_names=["EEG1"],
            fs=173.61,
            dataset_name="bonn",
        )

    # In-progress gauge decremented back to 0
    assert JOBS_IN_PROGRESS.get_value(labels={"execution_mode": "in_process"}) == 0.0

    # Total completed not incremented
    assert JOBS_COMPLETED_TOTAL.get_value(labels={"dataset": "bonn", "execution_mode": "in_process"}) == 0.0

    # Total failed incremented
    assert JOBS_FAILED_TOTAL.get_value(
        labels={
            "dataset": "bonn",
            "execution_mode": "in_process",
            "error_category": SAFE_ERROR_INFERENCE,
        }
    ) == 1.0

    # Failed duration recorded
    exec_stats = JOB_EXECUTION_SECONDS.get_stats(
        labels={"dataset": "bonn", "execution_mode": "in_process", "status": "failed"}
    )
    assert exec_stats["count"] == 1


@pytest.mark.asyncio
async def test_worker_task_telemetry_and_queue_wait(isolated_db, tmp_path, monkeypatch):
    """Test worker execution records queue wait, claim, staging load, and cleanup stages."""
    storage = LocalStorageBackend(str(tmp_path / "storage"))
    monkeypatch.setattr("app.services.storage.storage_backend", storage)

    job_id = f"job_dist_{uuid.uuid4().hex[:8]}"
    job = PredictionJob(
        id=job_id,
        tenant_id=TEST_TENANT_ID,
        status="Queued",
        detected_dataset="bonn",
    )
    isolated_db.add(job)
    isolated_db.commit()

    # Stage payload
    eeg_data = np.random.randn(1, 4097).astype(np.float32)
    staged_path = storage.save_window(
        job_id=job_id,
        eeg_data=eeg_data,
        channel_names=["EEG1"],
        fs=173.61,
        dataset_name="bonn",
    )

    worker_id = "test_worker_telemetry_01"
    simulated_enqueue_time = time.time() - 1.5

    result = await run_prediction_task(
        {},
        job_id=job_id,
        staged_path=staged_path,
        dataset_name="bonn",
        worker_id=worker_id,
        enqueued_at=simulated_enqueue_time,
    )
    assert result["status"] == "completed"

    # Queue wait observed
    wait_stats = QUEUE_WAIT_SECONDS.get_stats(labels={"dataset": "bonn"})
    assert wait_stats["count"] == 1
    assert wait_stats["sum"] >= 1.0

    # Worker claim stage observed
    claim_stats = PIPELINE_STAGE_SECONDS.get_stats(
        labels={"stage": "worker_claim", "dataset": "bonn", "status": "success"}
    )
    assert claim_stats["count"] == 1

    # Staging load stage observed
    load_stats = PIPELINE_STAGE_SECONDS.get_stats(
        labels={"stage": "staging_load", "dataset": "bonn", "status": "success"}
    )
    assert load_stats["count"] == 1

    # Cleanup stage observed
    cleanup_stats = PIPELINE_STAGE_SECONDS.get_stats(
        labels={"stage": "cleanup", "dataset": "bonn", "status": "success"}
    )
    assert cleanup_stats["count"] == 1

    # Distributed completed counter
    assert JOBS_COMPLETED_TOTAL.get_value(labels={"dataset": "bonn", "execution_mode": "distributed"}) == 1.0


@pytest.mark.asyncio
async def test_worker_claim_collision_telemetry(isolated_db):
    """Test claim collisions increment WORKER_CLAIM_FAILURES_TOTAL."""
    job_id = f"job_collision_{uuid.uuid4().hex[:8]}"
    now = datetime.datetime.utcnow()
    job = PredictionJob(
        id=job_id,
        tenant_id=TEST_TENANT_ID,
        status="Running",
        worker_id="active_owner_worker",
        lease_expires_at=now + datetime.timedelta(seconds=60),
        detected_dataset="bonn",
    )
    isolated_db.add(job)
    isolated_db.commit()

    result = await run_prediction_task(
        {},
        job_id=job_id,
        staged_path="dummy_path.npz",
        dataset_name="bonn",
        worker_id="intruder_worker",
    )
    assert result["status"] == "failed"

    assert WORKER_CLAIM_FAILURES_TOTAL.get_value() == 1.0
    claim_failed_stats = PIPELINE_STAGE_SECONDS.get_stats(
        labels={"stage": "worker_claim", "dataset": "bonn", "status": "failed"}
    )
    assert claim_failed_stats["count"] == 1


def test_stale_job_reaper_telemetry(isolated_db):
    """Test reaper increments JOBS_REAPED_TOTAL and JOBS_FAILED_TOTAL."""
    job_id = f"job_stale_{uuid.uuid4().hex[:8]}"
    past = datetime.datetime.utcnow() - datetime.timedelta(seconds=300)
    job = PredictionJob(
        id=job_id,
        tenant_id=TEST_TENANT_ID,
        status="Running",
        worker_id="dead_worker",
        lease_expires_at=past,
        detected_dataset="bonn",
    )
    isolated_db.add(job)
    isolated_db.commit()

    reaped = reap_stale_jobs(isolated_db)
    assert job_id in reaped

    assert JOBS_REAPED_TOTAL.get_value() == 1.0
    assert JOBS_FAILED_TOTAL.get_value(
        labels={
            "dataset": "bonn",
            "execution_mode": "distributed",
            "error_category": SAFE_ERROR_LEASE_EXPIRED,
        }
    ) == 1.0


def test_metrics_endpoint_exposition_and_separation(test_client):
    """Test GET /metrics returns Prometheus exposition format and does not collide with /api/v1/metrics."""
    # Observe some metrics
    JOBS_CREATED_TOTAL.inc(labels={"dataset": "bonn", "execution_mode": "in_process"})
    JOBS_REAPED_TOTAL.inc()

    res = test_client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    assert "version=0.0.4" in res.headers["content-type"]

    body = res.text
    assert "# HELP neuroaegis_jobs_created_total" in body
    assert '# TYPE neuroaegis_jobs_created_total counter' in body
    assert 'neuroaegis_jobs_created_total{dataset="bonn",execution_mode="in_process"} 1' in body
    assert 'neuroaegis_jobs_reaped_total 1' in body

    eval_res = test_client.get("/api/v1/metrics")
    assert eval_res.status_code == 200
    assert "application/json" in eval_res.headers["content-type"]
    data = eval_res.json()
    assert "average_metrics" in data or "metadata" in data


def test_v2_predict_queue_dispatch_failure_telemetry(test_client, auth_headers, monkeypatch, tmp_path):
    """Test distributed queue failure in v2 predict updates QUEUE_DISPATCH_FAILURES_TOTAL and JOBS_FAILED_TOTAL."""
    import json
    import tempfile
    from app.services.queue import QueueConnectionError
    from tests.test_api_v2_dispatch import make_synthetic_chbmit_edf_bytes

    monkeypatch.setattr(settings, "ENABLE_DISTRIBUTED_QUEUE", True)
    test_storage = LocalStorageBackend(storage_dir=str(tmp_path / "staging"))
    monkeypatch.setattr("app.api.v2.predict.storage_backend", test_storage)

    mock_queue = AsyncMock()
    mock_queue.enqueue_prediction_job.side_effect = QueueConnectionError("Redis cluster unreachable")
    monkeypatch.setattr("app.api.v2.predict.prediction_queue", mock_queue)

    edf_bytes = make_synthetic_chbmit_edf_bytes()
    res = test_client.post(
        "/api/v2/predict/",
        headers=auth_headers,
        data={
            "name": "Queue Fail Patient",
            "age": 45,
            "gender": "female",
            "weight": 65.0,
            "height": 165.0,
            "medical_history": json.dumps({}),
            "vital_signs": json.dumps({}),
        },
        files={"file": ("chb01_01.edf", io.BytesIO(edf_bytes), "application/octet-stream")},
    )

    assert res.status_code == 503
    assert QUEUE_DISPATCH_FAILURES_TOTAL.get_value(labels={"dataset": "chbmit"}) == 1.0
    assert JOBS_FAILED_TOTAL.get_value(
        labels={
            "dataset": "chbmit",
            "execution_mode": "distributed",
            "error_category": SAFE_ERROR_QUEUE_DISPATCH,
        }
    ) == 1.0

