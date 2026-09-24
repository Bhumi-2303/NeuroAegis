from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.core.config import settings
from tests.edf_fixture import write_synthetic_edf


# Use the real settings module while keeping this import-time database isolated.
settings_overrides = {
    "PROJECT_NAME": "NeuroAegis",
    "API_V1_STR": "/api/v1",
    "API_V2_STR": "/api/v2",
    "CORS_ORIGINS": [],
    "MAX_EEG_UPLOAD_BYTES": 50 * 1024 * 1024,
    "DATABASE_URL": "sqlite:///:memory:",
}

with patch.multiple(settings, **settings_overrides):
    from app.db.database import Base, engine, get_db
    from app.main import app
    Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_predict_endpoint_response_shape(tmp_path):
    fixture_path = write_synthetic_edf(
        tmp_path / "chb01_01.edf",
        channel_names=[f"Ch{i}" for i in range(23)],
    )
    file_content = fixture_path.read_bytes()
    
    with patch("app.services.dataset_detection.detector.dataset_detector") as mock_detector, \
         patch("app.api.v1.predict.process_and_save_prediction"), \
         patch("app.api.v1.predict.prediction_router") as mock_router:
        mock_router.is_loaded = True
        mock_router.get_available_models.return_value = {
            "chbmit": {"dataset_info": {"sampling_rate": 256.0, "window_length": 15360}},
            "bonn": {"dataset_info": {"sampling_rate": 173.61, "window_length": 4097}}
        }
        
        # Mock database session to prevent actual DB writes during test
        mock_db = MagicMock()
        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            mock_detector.detect.return_value = ("chbmit", 1.0, ["chbmit_channels"])

            response = client.post(
                "/api/v1/predict/",
                files={"file": ("chb01_01.edf", file_content, "application/edf")},
                data={"sampling_rate": 256.0, "dataset": "chbmit", "model": "lightgbm"}
            )

            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

            data = response.json()
            assert "job_id" in data
            assert data["detected_dataset"] == "chbmit"
            assert data["confidence"] == 1.0
            assert data["matched_rules"]
            assert any("Sampling rate matches" in rule for rule in data["matched_rules"])
            assert data["selected_model"] == "lightgbm"
            assert data["validation"]["validationStatus"] == "valid"
            assert data["validation"]["dataset"] == "chbmit"
        finally:
            app.dependency_overrides.clear()


def test_legacy_v2_endpoint_rejects_non_edf_uploads():
    response = client.post(
        "/api/v2/predict",
        data={
            "name": "Test Patient",
            "age": 30,
            "gender": "unknown",
            "weight": 70,
            "height": 170,
            "medical_history": "{}",
            "vital_signs": "{}",
        },
        files={"file": ("record.csv", b"x,y\n1,2\n", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["validation"]["errors"] == ["Only .edf files are supported"]
