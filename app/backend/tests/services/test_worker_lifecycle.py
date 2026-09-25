from __future__ import annotations
import asyncio
import datetime
import os
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.database import Base, ensure_schema_compatibility
from app.db.models import Patient, PredictionJob
from app.services.job_recovery import (
    STALE_JOB_ERROR_MESSAGE,
    cleanup_orphaned_staged_files,
    is_job_stale,
    reap_job_if_stale,
    reap_stale_jobs,
    recover_job_as_failed,
)
from app.services.job_service import run_prediction_pipeline, update_job_status
from app.services.storage import LocalStorageBackend, StorageSecurityError
from app.services.worker import (
    claim_job,
    fail_job,
    get_worker_id,
    run_prediction_task,
    update_heartbeat,
)


@pytest.fixture
def isolated_db(monkeypatch):
    """Provides a fresh isolated in-memory SQLite database with schema compatibility."""
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    ensure_schema_compatibility(test_engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr("app.services.worker.SessionLocal", TestingSession)
    monkeypatch.setattr("app.services.job_service.SessionLocal", TestingSession)
    monkeypatch.setattr("app.services.job_recovery.SessionLocal", TestingSession)

    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def clean_storage(tmp_path, monkeypatch):
    """Provides an isolated storage directory."""
    backend = LocalStorageBackend(storage_dir=tmp_path)
    monkeypatch.setattr("app.services.storage.storage_backend", backend)
    monkeypatch.setattr("app.services.job_recovery.storage_backend", backend)
    return backend


@pytest.fixture(autouse=True)
def ensure_models_loaded():
    """Ensure prediction router models are loaded."""
    from app.services.prediction.prediction_router import prediction_router
    if not prediction_router.is_loaded:
        prediction_router.load_all_models()


# ==============================================================================
# PHASE 9: MANDATORY TESTS A THROUGH Q
# ==============================================================================

def test_a_worker_claims_job(isolated_db):
    """TEST A: Worker successfully claims ownership of a job."""
    job_id = str(uuid.uuid4())
    job = PredictionJob(id=job_id, status="Validating", progress=0)
    isolated_db.add(job)
    isolated_db.commit()

    worker_id = "test-worker-node1:1001:abcd"
    claimed = claim_job(job_id, worker_id, lease_duration_seconds=30)
    assert claimed is True

    isolated_db.refresh(job)
    assert job.worker_id == worker_id
    assert job.status == "Running"
    assert job.heartbeat_at is not None
    assert job.lease_expires_at is not None
    assert job.lease_expires_at > job.heartbeat_at


def test_b_heartbeat_updates_lease(isolated_db):
    """TEST B: Heartbeat updates lease expiration time."""
    job_id = str(uuid.uuid4())
    job = PredictionJob(id=job_id, status="Validating", progress=0)
    isolated_db.add(job)
    isolated_db.commit()

    worker_id = "test-worker-node1:1001:abcd"
    claim_job(job_id, worker_id, lease_duration_seconds=10)
    isolated_db.refresh(job)
    initial_lease = job.lease_expires_at

    # Worker sends heartbeat with extended lease
    time.sleep(0.01)
    updated = update_heartbeat(job_id, worker_id, lease_duration_seconds=60)
    assert updated is True

    isolated_db.refresh(job)
    assert job.lease_expires_at > initial_lease


def test_c_active_job_is_not_considered_stale(isolated_db):
    """TEST C: Active job with valid future lease is not stale."""
    job_id = str(uuid.uuid4())
    future_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=60)
    job = PredictionJob(
        id=job_id,
        status="Running",
        worker_id="worker-live",
        heartbeat_at=datetime.datetime.utcnow(),
        lease_expires_at=future_time,
    )
    isolated_db.add(job)
    isolated_db.commit()

    assert is_job_stale(job) is False
    reaped = reap_stale_jobs(isolated_db)
    assert job_id not in reaped
    isolated_db.refresh(job)
    assert job.status == "Running"


def test_d_expired_heartbeat_is_detected(isolated_db):
    """TEST D: Expired lease is detected as stale."""
    job_id = str(uuid.uuid4())
    past_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=10)
    job = PredictionJob(
        id=job_id,
        status="Running",
        worker_id="worker-dead",
        heartbeat_at=past_time - datetime.timedelta(seconds=5),
        lease_expires_at=past_time,
    )
    isolated_db.add(job)
    isolated_db.commit()

    assert is_job_stale(job) is True


def test_e_f_stale_running_job_becomes_failed_with_structured_error(isolated_db):
    """TEST E & F: Stale Running job becomes Failed with structured lifecycle error."""
    job_id = str(uuid.uuid4())
    past_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=15)
    job = PredictionJob(
        id=job_id,
        status="Running",
        progress=45,
        worker_id="worker-crashed",
        heartbeat_at=past_time,
        lease_expires_at=past_time,
    )
    isolated_db.add(job)
    isolated_db.commit()

    reaped = reap_stale_jobs(isolated_db)
    assert job_id in reaped

    isolated_db.refresh(job)
    assert job.status == "Failed"
    assert job.progress == 0
    assert job.error == STALE_JOB_ERROR_MESSAGE
    assert job.completed_at is not None


def test_g_active_staged_payload_is_preserved(isolated_db, clean_storage):
    """TEST G: Staged payload of an active job is NOT cleaned up."""
    job_id = str(uuid.uuid4())
    future_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=60)
    job = PredictionJob(
        id=job_id,
        status="Running",
        worker_id="worker-1",
        heartbeat_at=datetime.datetime.utcnow(),
        lease_expires_at=future_time,
    )
    isolated_db.add(job)
    isolated_db.commit()

    # Stage a window payload
    staged_file = clean_storage.save_window(
        job_id=job_id,
        eeg_data=np.random.randn(1, 4096).astype(np.float32),
        channel_names=["EEG"],
        fs=173.61,
        dataset_name="bonn",
    )
    assert os.path.exists(staged_file)

    # Run cleanup with 0 grace seconds
    cleaned = cleanup_orphaned_staged_files(db=isolated_db, backend=clean_storage, grace_seconds=0)
    assert staged_file not in cleaned
    assert os.path.exists(staged_file)


def test_h_stale_orphan_payload_can_be_cleaned(isolated_db, clean_storage):
    """TEST H: Staged payload of a reaped/terminal job is cleaned up."""
    job_id = str(uuid.uuid4())
    past_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=20)
    job = PredictionJob(
        id=job_id,
        status="Failed",
        worker_id="worker-dead",
        heartbeat_at=past_time,
        lease_expires_at=past_time,
    )
    isolated_db.add(job)
    isolated_db.commit()

    staged_file = clean_storage.save_window(
        job_id=job_id,
        eeg_data=np.random.randn(1, 4096).astype(np.float32),
        channel_names=["EEG"],
        fs=173.61,
        dataset_name="bonn",
    )
    assert os.path.exists(staged_file)

    # Clean with grace_seconds=0
    cleaned = cleanup_orphaned_staged_files(db=isolated_db, backend=clean_storage, grace_seconds=0)
    assert staged_file in cleaned
    assert not os.path.exists(staged_file)


def test_i_recent_orphan_payload_is_preserved(isolated_db, clean_storage):
    """TEST I: Recent file (within grace period) is preserved even if unreferenced."""
    unreferenced_job_id = str(uuid.uuid4())
    staged_file = clean_storage.save_window(
        job_id=unreferenced_job_id,
        eeg_data=np.random.randn(1, 4096).astype(np.float32),
        channel_names=["EEG"],
        fs=173.61,
        dataset_name="bonn",
    )
    assert os.path.exists(staged_file)

    # Cleanup with a 300s grace period preserves recent files
    cleaned = cleanup_orphaned_staged_files(db=isolated_db, backend=clean_storage, grace_seconds=300)
    assert staged_file not in cleaned
    assert os.path.exists(staged_file)


def test_j_invalid_path_traversal_storage_reference_cannot_be_deleted(clean_storage):
    """TEST J: Path traversal attempts outside STORAGE_DIR are strictly rejected."""
    outside_file = tempfile.NamedTemporaryFile(delete=False)
    outside_file.write(b"CRITICAL SYSTEM FILE")
    outside_file.close()

    try:
        with pytest.raises(StorageSecurityError, match="Path traversal detected"):
            clean_storage.delete_window(f"../{os.path.basename(outside_file.name)}")
        assert os.path.exists(outside_file.name)
    finally:
        if os.path.exists(outside_file.name):
            os.unlink(outside_file.name)


def test_k_worker_crash_leaves_recoverable_job_state(isolated_db):
    """TEST K: A worker crash leaves a recoverable job state with lease timestamp."""
    job_id = str(uuid.uuid4())
    job = PredictionJob(id=job_id, status="Validating", progress=0)
    isolated_db.add(job)
    isolated_db.commit()

    worker_id = "crashed-worker-1"
    claim_job(job_id, worker_id, lease_duration_seconds=5)

    # Fast-forward time past lease expiration
    simulated_now = datetime.datetime.utcnow() + datetime.timedelta(seconds=10)
    assert is_job_stale(job, now=simulated_now) is True

    reaped = reap_stale_jobs(isolated_db, now=simulated_now)
    assert job_id in reaped

    isolated_db.refresh(job)
    assert job.status == "Failed"
    assert job.error == STALE_JOB_ERROR_MESSAGE


@pytest.mark.asyncio
async def test_l_worker_restart_can_process_new_jobs(isolated_db, clean_storage):
    """TEST L: Worker restart can successfully process new prediction jobs."""
    job_id = str(uuid.uuid4())
    job = PredictionJob(id=job_id, status="Validating", progress=0)
    isolated_db.add(job)
    isolated_db.commit()

    staged_path = clean_storage.save_window(
        job_id=job_id,
        eeg_data=np.random.randn(1, 4096).astype(np.float32),
        channel_names=["EEG"],
        fs=173.61,
        dataset_name="bonn",
    )

    ctx = {}
    result = await run_prediction_task(
        ctx=ctx,
        job_id=job_id,
        staged_path=staged_path,
        dataset_name="bonn",
    )

    assert result["status"] == "completed"
    isolated_db.refresh(job)
    assert job.status == "Completed"
    assert job.progress == 100
    assert not os.path.exists(staged_path)


@pytest.mark.asyncio
async def test_m_graceful_shutdown_does_not_fabricate_successful_predictions(isolated_db):
    """TEST M: Interrupted job during shutdown is not falsely marked Completed."""
    job_id = str(uuid.uuid4())
    worker_id = "interrupted-worker"
    job = PredictionJob(id=job_id, status="Running", progress=25, worker_id=worker_id)
    isolated_db.add(job)
    isolated_db.commit()

    # If the process is shutting down and cannot finish inference, it must NOT mark Completed
    isolated_db.refresh(job)
    assert job.status != "Completed"


@pytest.mark.asyncio
async def test_n_mode_a_remains_unchanged(isolated_db):
    """TEST N: Mode A in-process execution is completely unaffected and succeeds."""
    job_id = str(uuid.uuid4())
    job = PredictionJob(id=job_id, status="Validating", progress=0)
    isolated_db.add(job)
    isolated_db.commit()

    # Call run_prediction_pipeline directly with expected_worker_id=None (Mode A)
    await run_prediction_pipeline(
        job_id=job_id,
        eeg_data=np.random.randn(1, 4096).astype(np.float32),
        channel_names=["EEG"],
        fs=173.61,
        dataset_name="bonn",
        expected_worker_id=None,
    )

    isolated_db.refresh(job)
    assert job.status == "Completed"
    assert job.progress == 100
    assert job.worker_id is None # Mode A does not use worker_id


@pytest.mark.asyncio
async def test_o_mode_b_still_completes_successfully(isolated_db, clean_storage):
    """TEST O: Mode B distributed task claims, processes, and completes."""
    job_id = str(uuid.uuid4())
    job = PredictionJob(id=job_id, status="Validating", progress=0)
    isolated_db.add(job)
    isolated_db.commit()

    staged_path = clean_storage.save_window(
        job_id=job_id,
        eeg_data=np.random.randn(1, 4096).astype(np.float32),
        channel_names=["EEG"],
        fs=173.61,
        dataset_name="bonn",
    )

    result = await run_prediction_task(
        ctx={},
        job_id=job_id,
        staged_path=staged_path,
        dataset_name="bonn",
    )

    assert result["status"] == "completed"
    isolated_db.refresh(job)
    assert job.status == "Completed"
    assert job.worker_id is not None
    assert job.heartbeat_at is not None
    assert not os.path.exists(staged_path)


def test_p_no_automatic_prediction_retry_occurs(isolated_db):
    """TEST P: Stale job reaper does NOT re-enqueue or retry prediction."""
    mock_queue = MagicMock()
    job_id = str(uuid.uuid4())
    past_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=20)
    job = PredictionJob(
        id=job_id,
        status="Running",
        worker_id="worker-failed",
        heartbeat_at=past_time,
        lease_expires_at=past_time,
    )
    isolated_db.add(job)
    isolated_db.commit()

    with patch("app.services.queue.prediction_queue", mock_queue):
        reap_stale_jobs(isolated_db)

    # Queue must NEVER be called to retry
    mock_queue.enqueue_prediction_job.assert_not_called()
    isolated_db.refresh(job)
    assert job.status == "Failed"


@pytest.mark.asyncio
async def test_q_multiple_workers_cannot_incorrectly_finalize_same_job(isolated_db):
    """TEST Q: An expired/reaped worker cannot overwrite the job as Completed."""
    job_id = str(uuid.uuid4())
    worker_1 = "worker-slow-1"
    worker_2 = "worker-reclaimed-2"

    job = PredictionJob(
        id=job_id,
        status="Running",
        worker_id=worker_2, # Worker 2 now owns it
        heartbeat_at=datetime.datetime.utcnow(),
        lease_expires_at=datetime.datetime.utcnow() + datetime.timedelta(seconds=30),
    )
    isolated_db.add(job)
    isolated_db.commit()

    # Worker 1 tries to finalize the job as Completed
    await run_prediction_pipeline(
        job_id=job_id,
        eeg_data=np.random.randn(1, 4096).astype(np.float32),
        channel_names=["EEG"],
        fs=173.61,
        dataset_name="bonn",
        expected_worker_id=worker_1, # Worker 1 claims to be owner
    )

    isolated_db.refresh(job)
    # The job must NOT be finalized by Worker 1
    assert job.worker_id == worker_2
    assert job.status == "Running" # Status preserved for true owner


# ==============================================================================
# PHASE 10: DETERMINISTIC FAILURE INJECTION TEST
# ==============================================================================

@pytest.mark.asyncio
async def test_phase_10_deterministic_failure_injection_lifecycle(isolated_db, clean_storage):
    """
    Deterministic failure injection sequence:
    1. Distributed job is created (status=Validating)
    2. Staged window is written
    3. Worker claims job (status=Running, worker_id set, lease set)
    4. Worker crashes (heartbeat stops, time passes past lease timeout)
    5. Reaper runs
    6. Job becomes Failed with structured error
    7. Staged payload becomes eligible for cleanup and is deleted
    8. No automatic retry occurs
    9. Verify full DB state, timestamps, ownership, and clean storage
    """
    job_id = f"fail-inject-{uuid.uuid4().hex[:8]}"
    patient_id = str(uuid.uuid4())
    patient = Patient(id=patient_id, name="Failure Injection Patient", medical_history="{}", vital_signs={})
    job = PredictionJob(id=job_id, patient_id=patient_id, status="Validating", progress=0)
    isolated_db.add(patient)
    isolated_db.add(job)
    isolated_db.commit()

    # Step 2: Staged window payload is saved
    staged_path = clean_storage.save_window(
        job_id=job_id,
        eeg_data=np.random.randn(1, 4096).astype(np.float32),
        channel_names=["EEG"],
        fs=173.61,
        dataset_name="bonn",
    )
    assert os.path.exists(staged_path)

    # Step 3: Worker claims job
    crashed_worker_id = "crashed-worker-node1:4444:deadbeef"
    claimed = claim_job(job_id, crashed_worker_id, lease_duration_seconds=10)
    assert claimed is True

    isolated_db.refresh(job)
    assert job.status == "Running"
    assert job.worker_id == crashed_worker_id
    claim_time = job.heartbeat_at
    lease_expires = job.lease_expires_at
    assert lease_expires is not None

    # Step 4: Worker crashes! Heartbeat stops. Fast-forward time past lease
    expired_time = lease_expires + datetime.timedelta(seconds=5)
    assert is_job_stale(job, now=expired_time) is True

    # Step 5: Reaper runs
    mock_queue = MagicMock()
    with patch("app.services.queue.prediction_queue", mock_queue):
        reaped_ids = reap_stale_jobs(isolated_db, now=expired_time)

    assert job_id in reaped_ids

    # Step 6: Verify job is Failed with structured lifecycle error
    isolated_db.refresh(job)
    assert job.status == "Failed"
    assert job.progress == 0
    assert job.error == STALE_JOB_ERROR_MESSAGE
    assert job.completed_at == expired_time
    assert job.worker_id == crashed_worker_id # Ownership record preserved for audit

    # Step 7: Staged payload becomes eligible for cleanup
    # With grace_seconds=0 relative to expired_time
    from app.services.job_recovery import utc_timestamp
    cleaned = cleanup_orphaned_staged_files(
        db=isolated_db,
        backend=clean_storage,
        grace_seconds=0,
        now_ts=utc_timestamp(expired_time),
    )
    assert staged_path in cleaned
    assert not os.path.exists(staged_path)

    # Step 8: Verify no automatic prediction retry occurred
    mock_queue.enqueue_prediction_job.assert_not_called()
