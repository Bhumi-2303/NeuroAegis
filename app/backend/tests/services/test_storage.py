from __future__ import annotations
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.core.config import settings
from app.db.models import PredictionJob
from app.services.queue import InMemoryPredictionQueue, QueueConnectionError
from app.services.storage import (
    LocalStorageBackend,
    StorageError,
    StorageNotFoundError,
    StorageSecurityError,
)


@pytest.fixture
def temp_storage():
    """Provides an isolated temporary directory for storage backend testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        backend = LocalStorageBackend(storage_dir=tmp_dir)
        yield backend, Path(tmp_dir)


# ==============================================================================
# GROUP 1: Storage Backend Core Functionality & Atomic Writes
# ==============================================================================

def test_storage_atomic_write_creates_valid_npz(temp_storage):
    """Verify atomic write creates a complete, uncorrupted canonical .npz package."""
    backend, tmp_path = temp_storage
    job_id = str(uuid.uuid4())
    eeg_data = np.random.randn(23, 1000).astype(np.float32)
    channels = [f"EEG_{i+1}" for i in range(23)]
    fs = 256.0
    vis = {"channels": [{"name": "EEG_1", "samples": [0.1, 0.2]}]}

    staged_file = backend.save_window(
        job_id=job_id,
        eeg_data=eeg_data,
        channel_names=channels,
        fs=fs,
        dataset_name="chbmit",
        eeg_visualization=vis,
        metadata={"patient_id": "p-123"},
    )

    assert os.path.exists(staged_file)
    assert staged_file.endswith(f"{job_id}.npz")
    # Verify no dangling temporary files remain
    tmp_files = [f for f in os.listdir(tmp_path) if f.startswith(".tmp_")]
    assert len(tmp_files) == 0


def test_storage_load_window_retrieves_exact_data(temp_storage):
    """Verify load_window faithfully reconstructs all saved array and metadata fields."""
    backend, _ = temp_storage
    job_id = str(uuid.uuid4())
    eeg_data = np.random.randn(1, 4096).astype(np.float32)
    channels = ["EEG"]
    fs = 173.61
    vis = {"channels": [{"name": "EEG", "samples": [0.5]}], "samplingRate": 173.61}

    staged_file = backend.save_window(
        job_id=job_id,
        eeg_data=eeg_data,
        channel_names=channels,
        fs=fs,
        dataset_name="bonn",
        eeg_visualization=vis,
        metadata={"patient_id": "p-bonn"},
    )

    loaded_eeg, loaded_chs, loaded_fs, loaded_ds, loaded_vis, loaded_meta = backend.load_window(staged_file)

    np.testing.assert_array_almost_equal(loaded_eeg, eeg_data)
    assert loaded_chs == channels
    assert loaded_fs == fs
    assert loaded_ds == "bonn"
    assert loaded_vis == vis
    assert loaded_meta == {"patient_id": "p-bonn"}


def test_storage_delete_window_removes_file(temp_storage):
    """Verify delete_window cleanly unlinks the package and returns proper status."""
    backend, _ = temp_storage
    job_id = str(uuid.uuid4())
    eeg_data = np.zeros((1, 100), dtype=np.float32)

    staged_file = backend.save_window(
        job_id=job_id,
        eeg_data=eeg_data,
        channel_names=["CH1"],
        fs=100.0,
        dataset_name="bonn",
    )
    assert os.path.exists(staged_file)

    assert backend.delete_window(staged_file) is True
    assert not os.path.exists(staged_file)
    # Idempotent delete on non-existent file returns False
    assert backend.delete_window(staged_file) is False


# ==============================================================================
# GROUP 2: Security & Path Traversal Guards
# ==============================================================================

def test_storage_path_traversal_prohibited(temp_storage):
    """Verify that path traversal attempts raise StorageSecurityError."""
    backend, _ = temp_storage

    with pytest.raises(StorageSecurityError, match="Path traversal detected"):
        backend.load_window("../../etc/passwd")

    with pytest.raises(StorageSecurityError, match="Path traversal detected"):
        backend.delete_window("../../../sensitive_file")

    with pytest.raises(StorageSecurityError, match="Unsafe job_id format"):
        backend.save_window(
            job_id="../../traversal_id",
            eeg_data=np.zeros((1, 10)),
            channel_names=["CH"],
            fs=100.0,
            dataset_name="bonn",
        )


# ==============================================================================
# GROUP 3: Failure Handling & Atomic Cleanup
# ==============================================================================

def test_storage_atomic_write_cleans_up_on_failure(temp_storage):
    """Verify partial writes are removed when serialization or fsync fails."""
    backend, tmp_path = temp_storage
    job_id = str(uuid.uuid4())

    with patch("numpy.savez", side_effect=IOError("Disk write fault")):
        with pytest.raises(StorageError, match="Failed to save staged window package"):
            backend.save_window(
                job_id=job_id,
                eeg_data=np.zeros((1, 10)),
                channel_names=["CH"],
                fs=100.0,
                dataset_name="bonn",
            )

    # Verify no partial temporary files were left behind
    assert len(os.listdir(tmp_path)) == 0


def test_storage_missing_file_raises_storage_not_found_error(temp_storage):
    """Verify missing file raises StorageNotFoundError."""
    backend, _ = temp_storage
    with pytest.raises(StorageNotFoundError, match="not found"):
        backend.load_window("non_existent_job_123.npz")


def test_storage_corrupted_file_raises_storage_error(temp_storage):
    """Verify invalid/corrupt archive raises StorageError."""
    backend, tmp_path = temp_storage
    corrupt_file = tmp_path / "corrupt_job.npz"
    corrupt_file.write_bytes(b"CORRUPTED_BINARY_NOT_NPZ")

    with pytest.raises(StorageError, match="Failed to load or parse"):
        backend.load_window(str(corrupt_file))
