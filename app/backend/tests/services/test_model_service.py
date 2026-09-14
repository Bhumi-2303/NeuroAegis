import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock app.core.config to prevent ImportError
mock_config = MagicMock()
mock_config.settings = MagicMock()
sys.modules["app.core.config"] = mock_config

from app.schemas.prediction import ModelOutputSchema
from app.services.model_service import ModelService


@pytest.fixture
def mock_prediction_result():
    return {
        "modelName": "lightgbm",
        "prediction": {
            "label": "seizure",
            "probabilities": {"seizure": 0.85, "non_seizure": 0.15}
        },
        "confidence": {
            "value": 0.85,
            "band": "high"
        },
        "explanation": {
            "features": [
                {"featureName": "mean", "value": 0.5},
                {"featureName": "std", "value": 0.3}
            ],
            "baseValue": 0.1
        },
        "generatedAt": "2026-08-02T12:00:00Z"
    }

@patch("app.services.model_service.prediction_router")
def test_model_service_get_prediction(mock_prediction_router, mock_prediction_result):
    # Setup mock predictor
    mock_predictor = MagicMock()
    mock_predictor.get_prediction.return_value = mock_prediction_result
    mock_prediction_router.get_predictor.return_value = mock_predictor
    
    # Initialize service
    service = ModelService()
    
    # Dummy data
    eeg_data = np.random.randn(4097)
    channel_names = ["Ch0"]
    
    # Call method
    result = service.get_prediction(eeg_data, channel_names, fs=256.0)
    
    # Assertions
    mock_prediction_router.get_predictor.assert_called_once_with("bonn")
    mock_predictor.get_prediction.assert_called_once_with(eeg_data, channel_names, 256.0, model_name="lightgbm")
    
    assert isinstance(result, ModelOutputSchema)
    assert result.modelName == "lightgbm"
    assert result.prediction.label == "seizure"
    assert result.prediction.probabilities["seizure"] == 0.85
    assert result.confidence.value == 0.85
    assert result.confidence.band == "high"
    assert len(result.explanation.features) == 2
    assert result.generatedAt == "2026-08-02T12:00:00Z"

@patch("app.services.model_service.prediction_router")
def test_model_service_is_loaded(mock_prediction_router):
    mock_prediction_router.is_loaded = True
    service = ModelService()
    assert service.is_loaded is True

@patch("app.services.model_service.prediction_router")
def test_model_service_load_artifacts(mock_prediction_router):
    mock_prediction_router.load_all_models.return_value = True
    service = ModelService()
    assert service.load_artifacts() is True
    mock_prediction_router.load_all_models.assert_called_once()
