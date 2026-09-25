from __future__ import annotations
import io
import json
import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import mne
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import PredictionJob
from app.main import app
from app.services.queue import InMemoryPredictionQueue, QueueConnectionError
from app.services.storage import LocalStorageBackend, StorageError


from pathlib import Path
from tests.edf_fixture import write_synthetic_edf


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


@pytest.fixture(autouse=True)
def ensure_models_loaded():
    """Ensure prediction router models are loaded for testing API v2 dispatch."""
    from app.services.prediction.prediction_router import prediction_router
    if not prediction_router.is_loaded:
        prediction_router.load_all_models()


# ==============================================================================
# MODE A: Default In-Process Execution (ENABLE_DISTRIBUTED_QUEUE=False)
# ==============================================================================

def test_v2_predict_default_mode_uses_in_process_execution(monkeypatch):
    """
    Mandatory Regression Test (Prompt 7.3 Section 7):
    When ENABLE_DISTRIBUTED_QUEUE is False (default):
    - API v2 uses exact Prompt 6 in-process background execution
    - StorageBackend is NOT called
    - Redis Queue is NOT called
    """
    monkeypatch.setattr(settings, "ENABLE_DISTRIBUTED_QUEUE", False)

    mock_save_window = MagicMock()
    mock_enqueue = AsyncMock()
    monkeypatch.setattr("app.api.v2.predict.storage_backend.save_window", mock_save_window)
    monkeypatch.setattr("app.api.v2.predict.prediction_queue.enqueue_prediction_job", mock_enqueue)

    client = TestClient(app)
    edf_bytes = make_synthetic_chbmit_edf_bytes()

    response = client.post(
        "/api/v2/predict/",
        data={
            "name": "Jane Doe",
            "age": 30,
            "gender": "female",
            "weight": 60.0,
            "height": 165.0,
            "medical_history": json.dumps({"hypertension": False}),
            "vital_signs": json.dumps({"heart_rate": 72}),
        },
        files={"file": ("chb01_01.edf", io.BytesIO(edf_bytes), "application/octet-stream")},
    )

    assert response.status_code == 200
    res_data = response.json()
    assert "job_id" in res_data
    assert res_data["detected_dataset"] == "chbmit"

    # Verify that storage and queue were NOT touched
    mock_save_window.assert_not_called()
    mock_enqueue.assert_not_called()


# ==============================================================================
# MODE B: Controlled Distributed Queue Dispatch (ENABLE_DISTRIBUTED_QUEUE=True)
# ==============================================================================

@pytest.mark.asyncio
async def test_v2_predict_distributed_mode_stages_window_and_enqueues(monkeypatch):
    """
    Prompt 7.3 Section 4 & 8:
    When ENABLE_DISTRIBUTED_QUEUE is True:
    - Stages window via StorageBackend
    - Enqueues job via PredictionQueue with lightweight metadata ONLY
    - Preserves exact response schema
    """
    monkeypatch.setattr(settings, "ENABLE_DISTRIBUTED_QUEUE", True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_storage = LocalStorageBackend(storage_dir=tmp_dir)
        test_queue = InMemoryPredictionQueue()

        monkeypatch.setattr("app.api.v2.predict.storage_backend", test_storage)
        monkeypatch.setattr("app.api.v2.predict.prediction_queue", test_queue)

        client = TestClient(app)
        edf_bytes = make_synthetic_chbmit_edf_bytes()

        response = client.post(
            "/api/v2/predict/",
            data={
                "name": "Distributed Test Patient",
                "age": 42,
                "gender": "male",
                "weight": 75.0,
                "height": 178.0,
                "medical_history": json.dumps({"prior_seizure": True}),
                "vital_signs": json.dumps({"heart_rate": 80}),
            },
            files={"file": ("chb02_01.edf", io.BytesIO(edf_bytes), "application/octet-stream")},
        )

        assert response.status_code == 200
        res_data = response.json()
        job_id = res_data["job_id"]
        assert "patient_id" in res_data
        assert res_data["detected_dataset"] == "chbmit"

        # Verify staged window was written
        expected_staged_path = os.path.join(tmp_dir, f"{job_id}.npz")
        assert os.path.exists(expected_staged_path)

        # Verify queue contains the job with lightweight identifier and path reference ONLY
        enqueued = test_queue.get_enqueued_jobs()
        assert len(enqueued) == 1
        assert enqueued[0]["job_id"] == job_id
        assert Path(enqueued[0]["staged_path"]).resolve() == Path(expected_staged_path).resolve()
        assert enqueued[0]["dataset_name"] == "chbmit"
        assert enqueued[0]["medical_history"] == {"prior_seizure": True}


# ==============================================================================
# MODE B Failure Semantics (No Silent Fallback!)
# ==============================================================================

def test_v2_predict_distributed_mode_fails_safely_on_queue_failure(monkeypatch):
    """
    Prompt 7.3 Section 6 & 9:
    If Redis queue fails:
    - Mark DB job as Failed
    - Clean up staged storage file
    - Raise HTTP 503
    - NEVER execute in-process as a silent fallback
    """
    monkeypatch.setattr(settings, "ENABLE_DISTRIBUTED_QUEUE", True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_storage = LocalStorageBackend(storage_dir=tmp_dir)
        monkeypatch.setattr("app.api.v2.predict.storage_backend", test_storage)

        # Queue simulates broker connection failure
        mock_queue = AsyncMock()
        mock_queue.enqueue_prediction_job.side_effect = QueueConnectionError("Redis cluster unreachable")
        monkeypatch.setattr("app.api.v2.predict.prediction_queue", mock_queue)

        client = TestClient(app)
        edf_bytes = make_synthetic_chbmit_edf_bytes()

        response = client.post(
            "/api/v2/predict/",
            data={
                "name": "Failure Test Patient",
                "age": 50,
                "gender": "male",
                "weight": 80.0,
                "height": 180.0,
                "medical_history": json.dumps({}),
                "vital_signs": json.dumps({}),
            },
            files={"file": ("chb03_01.edf", io.BytesIO(edf_bytes), "application/octet-stream")},
        )

        assert response.status_code == 503
        assert "Distributed queue dispatch failed" in response.json()["detail"]

        # Verify no staged files linger in storage directory
        assert len(os.listdir(tmp_dir)) == 0


def test_v2_predict_distributed_mode_fails_safely_on_storage_failure(monkeypatch):
    """
    Prompt 7.3 Section 6 & 9:
    If Storage write fails:
    - Raise HTTP 503
    - Mark DB job as Failed
    - Do NOT call queue
    - Do NOT fall back to in-process execution
    """
    monkeypatch.setattr(settings, "ENABLE_DISTRIBUTED_QUEUE", True)

    mock_storage = MagicMock()
    mock_storage.save_window.side_effect = StorageError("Shared volume out of disk space")
    mock_queue = AsyncMock()

    monkeypatch.setattr("app.api.v2.predict.storage_backend", mock_storage)
    monkeypatch.setattr("app.api.v2.predict.prediction_queue", mock_queue)

    client = TestClient(app)
    edf_bytes = make_synthetic_chbmit_edf_bytes()

    response = client.post(
        "/api/v2/predict/",
        data={
            "name": "Storage Failure Patient",
            "age": 55,
            "gender": "female",
            "weight": 65.0,
            "height": 160.0,
            "medical_history": json.dumps({}),
            "vital_signs": json.dumps({}),
        },
        files={"file": ("chb04_01.edf", io.BytesIO(edf_bytes), "application/octet-stream")},
    )

    assert response.status_code == 503
    assert "Shared volume out of disk space" in response.json()["detail"]
    mock_queue.enqueue_prediction_job.assert_not_called()
