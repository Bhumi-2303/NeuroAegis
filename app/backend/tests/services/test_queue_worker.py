from __future__ import annotations
import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.database import Base
from app.db.models import Patient, PredictionJob
from app.services.queue import (
    InMemoryPredictionQueue,
    QueueConnectionError,
    QueuePayloadError,
    RedisPredictionQueue,
)
from app.services.worker import (
    WorkerSettings,
    load_staged_payload,
    run_prediction_task,
    shutdown,
    startup,
)


@pytest.fixture
def isolated_db(monkeypatch):
    """Provides a fresh isolated in-memory SQLite database for test execution."""
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr("app.services.worker.SessionLocal", TestingSession)
    monkeypatch.setattr("app.services.job_service.SessionLocal", TestingSession)

    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(autouse=True)
def ensure_models_loaded():
    """Ensure prediction router models are loaded for testing worker execution."""
    from app.services.prediction.prediction_router import prediction_router
    if not prediction_router.is_loaded:
        prediction_router.load_all_models()


@pytest.fixture
def staged_npz_payload(tmp_path, monkeypatch):
    """Generates a synthetic 1-channel Bonn-compatible .npz fixture."""
    from app.services.storage import storage_backend
    monkeypatch.setattr(storage_backend, "storage_dir", tmp_path)
    job_id = f"test-staged-{uuid.uuid4().hex[:8]}"
    eeg_data = np.random.randn(1, 4096).astype(np.float32)
    path = storage_backend.save_window(
        job_id=job_id,
        eeg_data=eeg_data,
        channel_names=["EEG"],
        fs=173.61,
        dataset_name="bonn",
    )
    yield path
    storage_backend.delete_window(path)


# ==============================================================================
# GROUP A: Queue Abstraction Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_in_memory_queue_enqueue_preserves_job_id_and_metadata():
    """Verify in-memory queue stores job metadata and identifiers correctly."""
    queue = InMemoryPredictionQueue()
    job_id = str(uuid.uuid4())
    staged_path = "/app/storage/test_123.npz"

    result = await queue.enqueue_prediction_job(
        job_id=job_id,
        staged_path=staged_path,
        dataset_name="chbmit",
        medical_history={"hypertension": True},
    )

    assert result == f"mock-{job_id}"
    enqueued = queue.get_enqueued_jobs()
    assert len(enqueued) == 1
    assert enqueued[0]["job_id"] == job_id
    assert enqueued[0]["staged_path"] == staged_path
    assert enqueued[0]["dataset_name"] == "chbmit"
    assert enqueued[0]["medical_history"] == {"hypertension": True}


@pytest.mark.asyncio
async def test_queue_rejects_numpy_arrays():
    """Architectural Rule: Queue MUST reject NumPy arrays to prevent memory bloat in Redis."""
    queue = InMemoryPredictionQueue()
    job_id = str(uuid.uuid4())
    raw_array = np.zeros((23, 1000))

    with pytest.raises(QueuePayloadError, match="NumPy arrays must not be sent via Redis"):
        await queue.enqueue_prediction_job(
            job_id=job_id,
            staged_path="/app/storage/valid.npz",
            eeg_data=raw_array,
        )


@pytest.mark.asyncio
async def test_queue_rejects_large_binary_data():
    """Architectural Rule: Queue MUST reject raw binary buffers exceeding 1KB."""
    queue = InMemoryPredictionQueue()
    job_id = str(uuid.uuid4())
    raw_buffer = bytes(5000)

    with pytest.raises(QueuePayloadError, match="Large binary buffers"):
        await queue.enqueue_prediction_job(
            job_id=job_id,
            staged_path="/app/storage/valid.npz",
            raw_eeg=raw_buffer,
        )


@pytest.mark.asyncio
async def test_queue_rejects_empty_job_id_and_path():
    """Ensure invalid identifier parameters are caught before dispatch."""
    queue = InMemoryPredictionQueue()

    with pytest.raises(QueuePayloadError, match="job_id must be a non-empty string"):
        await queue.enqueue_prediction_job(job_id="", staged_path="/app/storage/valid.npz")

    with pytest.raises(QueuePayloadError, match="staged_path must be a valid path/key"):
        await queue.enqueue_prediction_job(job_id="valid-id", staged_path="")


@pytest.mark.asyncio
async def test_redis_queue_builds_correct_enqueue_call():
    """Verify RedisPredictionQueue properly calls ARQ pool.enqueue_job with expected schema."""
    mock_pool = AsyncMock()
    mock_arq_job = MagicMock()
    mock_arq_job.job_id = "arq-task-999"
    mock_pool.enqueue_job.return_value = mock_arq_job

    queue = RedisPredictionQueue(redis_url="redis://localhost:6379/0", queue_name="test:queue")
    queue._pool = mock_pool

    job_id = str(uuid.uuid4())
    result = await queue.enqueue_prediction_job(
        job_id=job_id,
        staged_path="/app/storage/job.npz",
        dataset_name="bonn",
    )

    assert result == "arq-task-999"
    mock_pool.enqueue_job.assert_called_once_with(
        "run_prediction_task",
        job_id=job_id,
        staged_path="/app/storage/job.npz",
        dataset_name="bonn",
        medical_history=None,
        _queue_name="test:queue",
    )


@pytest.mark.asyncio
async def test_redis_queue_raises_queue_connection_error_on_network_failure():
    """Verify connection failures to Redis broker raise QueueConnectionError without silent fallback."""
    queue = RedisPredictionQueue(redis_url="redis://invalid-unreachable-host:9999/0")

    with patch("app.services.queue.create_pool", side_effect=OSError("DNS resolution failed")):
        with pytest.raises(QueueConnectionError, match="Could not connect to Redis broker"):
            await queue.enqueue_prediction_job(
                job_id="test-job",
                staged_path="/app/storage/test.npz",
            )


# ==============================================================================
# GROUP B: Worker Foundation & Pipeline Reuse Tests
# ==============================================================================

def test_worker_settings_loads_correct_configuration():
    """Verify ARQ WorkerSettings exposes the configured queue name, timeouts, and entrypoint."""
    assert len(WorkerSettings.functions) == 1
    assert WorkerSettings.functions[0] == run_prediction_task
    assert WorkerSettings.queue_name == settings.QUEUE_NAME
    assert WorkerSettings.max_jobs == settings.WORKER_MAX_JOBS
    assert WorkerSettings.job_timeout == settings.JOB_TIMEOUT_SECONDS
    assert WorkerSettings.max_tries == 1
    assert WorkerSettings.retry_jobs is False


@pytest.mark.asyncio
async def test_worker_startup_and_shutdown_hooks():
    """Verify worker lifecycle hooks execute cleanly without leaking state."""
    ctx = {}
    with patch("app.services.worker.prediction_router.load_all_models", return_value=True):
        await startup(ctx)
        assert "prediction_router" in ctx
        await shutdown(ctx)


@pytest.mark.asyncio
async def test_worker_task_reuses_prediction_pipeline_with_staged_npz(isolated_db, staged_npz_payload):
    """Verify worker task executes inference and completes the DB job using existing pipeline."""
    job_id = str(uuid.uuid4())
    patient_id = str(uuid.uuid4())

    patient = Patient(id=patient_id, name="Test Patient", medical_history="{}", vital_signs={})
    job = PredictionJob(id=job_id, patient_id=patient_id, status="Pending", progress=0)
    isolated_db.add(patient)
    isolated_db.add(job)
    isolated_db.commit()

    ctx = {}
    result = await run_prediction_task(
        ctx=ctx,
        job_id=job_id,
        staged_path=staged_npz_payload,
        dataset_name="bonn",
    )

    assert result["status"] == "completed"
    assert result["job_id"] == job_id

    updated_job = isolated_db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
    assert updated_job.status == "Completed"
    assert updated_job.progress == 100
    assert updated_job.prediction_label in ["seizure", "non_seizure"]
    assert updated_job.probability_seizure is not None
    assert updated_job.eeg_visualization is not None


# ==============================================================================
# GROUP C: Failure Behavior & Exception Safety Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_worker_task_handles_missing_staged_payload_gracefully(isolated_db, tmp_path, monkeypatch):
    """Verify worker fails job gracefully if staged payload file is missing."""
    from app.services.storage import storage_backend
    monkeypatch.setattr(storage_backend, "storage_dir", tmp_path)
    job_id = str(uuid.uuid4())
    patient_id = str(uuid.uuid4())

    patient = Patient(id=patient_id, name="Test Patient", medical_history="{}", vital_signs={})
    job = PredictionJob(id=job_id, patient_id=patient_id, status="Pending", progress=0)
    isolated_db.add(patient)
    isolated_db.add(job)
    isolated_db.commit()

    non_existent_path = str(tmp_path / "non_existent_file_9999.npz")
    result = await run_prediction_task(
        ctx={},
        job_id=job_id,
        staged_path=non_existent_path,
        dataset_name="bonn",
    )

    assert result["status"] == "failed"
    assert "not exist" in result["error"].lower() or "not found" in result["error"].lower()

    updated_job = isolated_db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
    assert updated_job.status == "Failed"
    assert updated_job.progress == 0
    assert "not exist" in updated_job.error.lower() or "not found" in updated_job.error.lower()


@pytest.mark.asyncio
async def test_worker_task_handles_missing_database_job_gracefully(isolated_db, tmp_path, monkeypatch):
    """Verify worker does not crash if job_id does not exist in the database."""
    from app.services.storage import storage_backend
    monkeypatch.setattr(storage_backend, "storage_dir", tmp_path)
    non_existent_job_id = str(uuid.uuid4())
    result = await run_prediction_task(
        ctx={},
        job_id=non_existent_job_id,
        staged_path=str(tmp_path / "dummy.npz"),
    )

    assert result["status"] == "failed"
    assert "not found in database" in result["error"]


@pytest.mark.asyncio
async def test_worker_task_handles_corrupt_payload_gracefully(isolated_db, tmp_path, monkeypatch):
    """Verify worker fails job gracefully if staged payload is corrupted."""
    from app.services.storage import storage_backend
    monkeypatch.setattr(storage_backend, "storage_dir", tmp_path)
    job_id = str(uuid.uuid4())
    patient_id = str(uuid.uuid4())

    patient = Patient(id=patient_id, name="Test Patient", medical_history="{}", vital_signs={})
    job = PredictionJob(id=job_id, patient_id=patient_id, status="Pending", progress=0)
    isolated_db.add(patient)
    isolated_db.add(job)
    isolated_db.commit()

    corrupt_path = tmp_path / f"corrupt_{uuid.uuid4().hex[:8]}.npz"
    corrupt_path.write_bytes(b"NOT_A_VALID_NUMPY_FILE_CORRUPT")

    try:
        result = await run_prediction_task(
            ctx={},
            job_id=job_id,
            staged_path=str(corrupt_path),
            dataset_name="bonn",
        )

        assert result["status"] == "failed"
        updated_job = isolated_db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        assert updated_job.status == "Failed"
    finally:
        if corrupt_path.exists():
            corrupt_path.unlink()


@pytest.mark.asyncio
async def test_worker_task_handles_pipeline_exception_gracefully(isolated_db, staged_npz_payload):
    """Verify worker catches unhandled exceptions from the inference pipeline and does not crash."""
    job_id = str(uuid.uuid4())
    patient_id = str(uuid.uuid4())

    patient = Patient(id=patient_id, name="Test Patient", medical_history="{}", vital_signs={})
    job = PredictionJob(id=job_id, patient_id=patient_id, status="Pending", progress=0)
    isolated_db.add(patient)
    isolated_db.add(job)
    isolated_db.commit()

    with patch("app.services.worker.run_prediction_pipeline", side_effect=RuntimeError("Pipeline internal crash")):
        result = await run_prediction_task(
            ctx={},
            job_id=job_id,
            staged_path=staged_npz_payload,
            dataset_name="bonn",
        )

        assert result["status"] == "failed"
        assert "Pipeline internal crash" in result["error"]

        updated_job = isolated_db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        assert updated_job.status == "Failed"
        assert "Pipeline internal crash" in updated_job.error
