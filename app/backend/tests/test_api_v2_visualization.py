from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db.database import Base, SessionLocal, engine
from app.db.models import Patient, PredictionJob
from app.main import app
from tests.edf_fixture import write_synthetic_edf


@pytest.fixture(scope="module")
def client():
    app.dependency_overrides.clear()
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_v2_predict_rejects_non_edf(client: TestClient):
    """API v2 must reject non-EDF files with 400 and structured validation detail."""
    for url in ["/api/v2/predict", "/api/v2/predict/"]:
        response = client.post(
            url,
            data={
                "name": "Jane Doe",
                "age": 28,
                "gender": "female",
                "weight": 60.0,
                "height": 165.0,
                "medical_history": "{}",
                "vital_signs": "{}",
            },
            files={"file": ("recording.csv", b"timestamp,ch1\n0,1", "text/csv")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "validation" in data.get("detail", {})
        assert data["detail"]["validation"]["validationStatus"] == "invalid"
        assert any("Only .edf files" in err for err in data["detail"]["validation"]["errors"])


def test_v2_predict_and_report_parity_with_chbmit(client: TestClient, tmp_path: Path):
    """API v2 must generate, persist, and serve bounded EEG visualization matching v1 parity."""
    # Standard 23 CHB-MIT channels
    chbmit_channels = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9",
        "FT9-FT10", "FT10-T8", "T8-P8-1"
    ]
    edf_path = write_synthetic_edf(
        tmp_path / "chb01_01.edf",
        channel_names=chbmit_channels,
        sampling_rate=256,
        duration_seconds=10,
    )

    with open(edf_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v2/predict/",
        data={
            "name": "Alex Patient",
            "age": 35,
            "gender": "male",
            "weight": 75.0,
            "height": 180.0,
            "medical_history": "{}",
            "vital_signs": "{}",
            "sampling_rate": 256.0,
        },
        files={"file": ("chb01_01.edf", file_bytes, "application/octet-stream")},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert "job_id" in data
    assert "patient_id" in data
    assert data["detected_dataset"] == "chbmit"
    assert data["validation"]["validationStatus"] == "valid"

    job_id = data["job_id"]

    # Poll status until Completed (or check up to 10s)
    status_data = None
    for _ in range(50):
        res = client.get(f"/api/v2/predict/status/{job_id}")
        assert res.status_code == 200
        status_data = res.json()
        if status_data["status"] == "Completed":
            break
        import time
        time.sleep(0.1)

    assert status_data is not None
    assert status_data["status"] == "Completed"
    assert status_data["progress"] == 100
    assert status_data["datasetName"] == "chbmit"

    # Verify eeg_visualization parity
    assert "result" in status_data
    result = status_data["result"]
    assert "eeg_visualization" in result
    viz = result["eeg_visualization"]
    assert viz is not None

    # Check channels: exactly 23 CHB-MIT channels
    assert viz["totalChannels"] == 23
    assert viz["eegChannelCount"] == 23
    assert len(viz["channels"]) == 23

    # Check bounded sample count (<= 2000 points per channel)
    assert viz["visualizationSampleCount"] <= 2000
    for ch in viz["channels"]:
        assert len(ch["samples"]) <= 2000
        assert ch["unit"] == "uV"
        assert ch["samplingRate"] > 0

    # Check real timestamps
    assert viz["durationSeconds"] == 10.0
    assert viz["timeStartSeconds"] == 0.0
    assert pytest.approx(viz["timeEndSeconds"], abs=0.05) == 10.0

    # Check annotation & activity structures
    assert "hasSeizureAnnotations" in viz
    assert "annotationStatus" in viz
    assert "referenceAvailable" in viz
    assert "channelActivityAvailable" in viz
    assert isinstance(viz["channelActivity"], list)

    # Verify /jobs/{job_id} alias endpoint returns identical visualization
    job_alias_res = client.get(f"/api/v2/jobs/{job_id}")
    assert job_alias_res.status_code == 200
    alias_data = job_alias_res.json()
    assert alias_data["result"]["eeg_visualization"] == viz

    # Verify /report/{job_id} endpoint returns the same persisted visualization
    report_res = client.get(f"/api/v2/report/{job_id}")
    assert report_res.status_code == 200
    report_data = report_res.json()
    assert "job" in report_data
    assert "patient" in report_data
    assert report_data["job"]["eeg_visualization"] == viz
    assert report_data["patient"]["name"] == "Alex Patient"


def test_v2_predict_dynamic_channels_above_23(client: TestClient, tmp_path: Path):
    """API v2 must support dynamic channel counts beyond 23 without artificial ceiling."""
    # 26 channels (e.g. high-density montage)
    custom_channels = [f"EEG-{i+1}" for i in range(26)]
    edf_path = write_synthetic_edf(
        tmp_path / "custom_26ch.edf",
        channel_names=custom_channels,
        sampling_rate=256,
        duration_seconds=5,
    )

    with open(edf_path, "rb") as f:
        file_bytes = f.read()

    # Pre-insert a completed job with 26 channels to verify visualization handling
    db = SessionLocal()
    job_id = str(uuid.uuid4())
    patient_id = str(uuid.uuid4())
    try:
        from app.services.eeg_visualization import build_eeg_visualization_from_raw
        import mne
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        try:
            viz = build_eeg_visualization_from_raw(
                raw,
                file_name="custom_26ch.edf",
                file_size_bytes=len(file_bytes),
                dataset="unknown",
            )
        finally:
            raw.close()

        patient = Patient(
            id=patient_id,
            name="Dynamic Patient",
            age=40,
            gender="other",
            weight=68.0,
            height=172.0,
            medical_history="{}",
            vital_signs={},
        )
        job = PredictionJob(
            id=job_id,
            patient_id=patient_id,
            status="Completed",
            progress=100,
            detected_dataset="unknown",
            detection_confidence=0.5,
            selected_model="lightgbm",
            prediction_label="non_seizure",
            probability_seizure=0.05,
            confidence_band="high",
            eeg_visualization=viz,
        )
        db.add(patient)
        db.add(job)
        db.commit()

        # Query via status endpoint
        res = client.get(f"/api/v2/predict/status/{job_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["result"]["eeg_visualization"]["totalChannels"] == 26
        assert data["result"]["eeg_visualization"]["eegChannelCount"] == 26
        assert len(data["result"]["eeg_visualization"]["channels"]) == 26

        # Query via report endpoint
        rep = client.get(f"/api/v2/report/{job_id}")
        assert rep.status_code == 200
        rep_data = rep.json()
        assert rep_data["job"]["eeg_visualization"]["totalChannels"] == 26

    finally:
        db.close()


def test_v2_status_and_report_404_and_failed_job(client: TestClient):
    """API v2 must return 404 for missing jobs and surface error message on Failed jobs."""
    missing_id = str(uuid.uuid4())
    assert client.get(f"/api/v2/predict/status/{missing_id}").status_code == 404
    assert client.get(f"/api/v2/jobs/{missing_id}").status_code == 404
    assert client.get(f"/api/v2/report/{missing_id}").status_code == 404

    # Create a Failed job
    db = SessionLocal()
    job_id = str(uuid.uuid4())
    patient_id = str(uuid.uuid4())
    try:
        patient = Patient(
            id=patient_id,
            name="Failed Case",
            age=50,
            gender="male",
            weight=80.0,
            height=175.0,
            medical_history="{}",
            vital_signs={},
        )
        job = PredictionJob(
            id=job_id,
            patient_id=patient_id,
            status="Failed",
            progress=0,
            error="Signal corrupted beyond recovery",
        )
        db.add(patient)
        db.add(job)
        db.commit()

        res = client.get(f"/api/v2/predict/status/{job_id}")
        assert res.status_code == 200
        assert res.json()["status"] == "Failed"
        assert res.json()["error"] == "Signal corrupted beyond recovery"

        rep = client.get(f"/api/v2/report/{job_id}")
        assert rep.status_code == 200
        assert rep.json()["job"]["status"] == "Failed"
        assert rep.json()["job"]["error"] == "Signal corrupted beyond recovery"
    finally:
        db.close()
