from __future__ import annotations

import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.database import Base, SessionLocal, engine, get_db
from app.db.models import PredictionJob
from app.main import app
from app.services.edf_validation import EDFValidationService
from tests.edf_fixture import write_synthetic_edf


@pytest.fixture(scope="module")
def client():
    app.dependency_overrides.clear()
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_non_edf_upload_rejected(client: TestClient):
    """Phase 4.2: Non-EDF files must be rejected immediately with 400."""
    response = client.post(
        "/api/v1/predict/",
        files={"file": ("recording.csv", b"timestamp,ch1,ch2\n0,1,2", "text/csv")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "validation" in data.get("detail", {})
    assert data["detail"]["validation"]["validationStatus"] == "invalid"
    assert any("Only .edf files" in err for err in data["detail"]["validation"]["errors"])


def test_empty_edf_upload_rejected(client: TestClient):
    """Phase 4.2: 0-byte EDF files must be rejected with 400."""
    response = client.post(
        "/api/v1/predict/",
        files={"file": ("empty.edf", b"", "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "validation" in data.get("detail", {})
    assert data["detail"]["validation"]["validationStatus"] == "invalid"


def test_corrupt_edf_upload_rejected(client: TestClient):
    """Phase 4.2: Corrupted binary files with .edf extension must be rejected."""
    corrupt_bytes = b"NOT_A_REAL_EDF_HEADER" + b"\x00" * 1024
    response = client.post(
        "/api/v1/predict/",
        files={"file": ("corrupt.edf", corrupt_bytes, "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["validation"]["validationStatus"] == "invalid"


def test_missing_job_returns_404(client: TestClient):
    """Phase 4.2: Non-existent job query must return 404."""
    random_job_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/jobs/{random_job_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_failed_job_surfaces_error(client: TestClient):
    """Phase 4.2: Failed jobs must surface the failure reason without crashing."""
    db = SessionLocal()
    job_id = str(uuid.uuid4())
    try:
        job = PredictionJob(
            id=job_id,
            status="Failed",
            progress=0,
            error="Signal corruption detected during GNN inference",
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    response = client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Failed"
    assert data["error"] == "Signal corruption detected during GNN inference"


def test_validation_performance_on_real_chbmit_files():
    """Phase 4.4: Validation must be header-only without preloading, completing in under 2s."""
    chb01_03_path = Path("/Volumes/BLACK-BOX/NeuroAegis/CHB-MIT Dataset/chb01/chb01_03.edf")
    if not chb01_03_path.exists():
        pytest.skip("Local CHB-MIT dataset not found")

    validator = EDFValidationService()

    start_time = time.perf_counter()
    result = validator.validate_file(chb01_03_path, file_name=chb01_03_path.name)
    duration = time.perf_counter() - start_time

    assert result.validation_status == "valid"
    assert result.dataset == "chbmit"
    assert result.sampling_rate == 256.0
    assert result.total_channels == 23
    assert duration < 2.0, f"Validation took {duration:.3f}s, expected < 2.0s for header-only read"


def test_large_file_validation_performance():
    """Phase 4.4: 177MB EDF must validate in under 2s without memory preloading."""
    chb04_27_path = Path("/Volumes/BLACK-BOX/NeuroAegis/CHB-MIT Dataset/chb04/chb04_27.edf")
    if not chb04_27_path.exists():
        pytest.skip("Local large CHB-MIT dataset not found")

    validator = EDFValidationService()

    start_time = time.perf_counter()
    result = validator.validate_file(chb04_27_path, file_name=chb04_27_path.name)
    duration = time.perf_counter() - start_time

    assert result.validation_status == "valid"
    assert result.file_size_bytes == 177_285_376
    assert duration < 2.0, f"177MB validation took {duration:.3f}s, expected < 2.0s"


def test_stale_state_job_isolation():
    """Phase 4.3: Sequential jobs must maintain strict isolation in the database."""
    db = SessionLocal()
    job_a_id = str(uuid.uuid4())
    job_b_id = str(uuid.uuid4())

    try:
        job_a = PredictionJob(
            id=job_a_id,
            status="Completed",
            progress=100,
            prediction_label="seizure",
            probability_seizure=0.88,
            confidence_band="high",
            detected_dataset="chbmit",
        )
        job_b = PredictionJob(
            id=job_b_id,
            status="Completed",
            progress=100,
            prediction_label="non_seizure",
            probability_seizure=0.12,
            confidence_band="high",
            detected_dataset="bonn",
        )
        db.add(job_a)
        db.add(job_b)
        db.commit()

        # Query A
        query_a = db.query(PredictionJob).filter(PredictionJob.id == job_a_id).first()
        assert query_a.prediction_label == "seizure"
        assert query_a.probability_seizure == 0.88
        assert query_a.detected_dataset == "chbmit"

        # Query B
        query_b = db.query(PredictionJob).filter(PredictionJob.id == job_b_id).first()
        assert query_b.prediction_label == "non_seizure"
        assert query_b.probability_seizure == 0.12
        assert query_b.detected_dataset == "bonn"

        # Neither contaminated the other
        assert query_a.id != query_b.id
        assert query_a.prediction_label != query_b.prediction_label
    finally:
        db.close()
