import sys
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi.testclient import TestClient

# Bypass the config module naming conflict
mock_settings = MagicMock()
mock_settings.PROJECT_NAME = "NeuroAegis"
mock_settings.API_V1_STR = "/api/v1"
mock_settings.API_V2_STR = "/api/v2"
mock_settings.CORS_ORIGINS = []
mock_settings.MAX_UPLOAD_SIZE = 50 * 1024 * 1024
mock_settings.DATABASE_URL = "sqlite:///:memory:"
mock_config = MagicMock()
mock_config.settings = mock_settings
sys.modules['app.core.config'] = mock_config

with patch("app.services.prediction.prediction_router.prediction_router") as mock_router:
    mock_router.is_loaded = True
    mock_router.get_available_models.return_value = {
        "chbmit": {"dataset_info": {"sampling_rate": 256.0, "window_length": 15360}},
        "bonn": {"dataset_info": {"sampling_rate": 173.61, "window_length": 4097}}
    }
    from app.db.database import Base, engine, get_db
    from app.main import app
    Base.metadata.create_all(bind=engine)
    
client = TestClient(app)

def test_predict_endpoint_response_shape():
    # Create a dummy CSV file that looks like EEG data
    df = pd.DataFrame({"Fp1": [0.1, 0.2, 0.3], "Fp2": [0.2, 0.3, 0.4]})
    file_content = df.to_csv(index=False).encode('utf-8')
    
    with patch("app.services.dataset_detection.detector.dataset_detector") as mock_detector, \
         patch("app.api.v1.predict.process_and_save_prediction"):
        
        # Mock database session to prevent actual DB writes during test
        mock_db = MagicMock()
        app.dependency_overrides[get_db] = lambda: mock_db
        
        mock_detector.detect.return_value = ("chbmit", 1.0, ["chbmit_channels"])
        
        response = client.post(
            "/api/v1/predict/",
            files={"file": ("dummy.csv", file_content, "text/csv")},
            data={"sampling_rate": 256.0, "dataset": "chbmit", "model": "lightgbm"}
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "job_id" in data
        assert data["detected_dataset"] == "chbmit"
        assert data["confidence"] == 1.0
        assert data["matched_rules"] == [] or data["matched_rules"] == ["chbmit_channels"]
        assert data["selected_model"] == "lightgbm"
