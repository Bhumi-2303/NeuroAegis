import pytest
import numpy as np
import pandas as pd
from app.services.prediction.prediction_router import prediction_router

@pytest.fixture(scope="module")
def load_models():
    # Ensure models are loaded before tests
    prediction_router.load_all_models()
    return prediction_router

def test_bonn_inference(load_models):
    predictor = load_models.get_predictor("bonn")
    # Mock Bonn dataset (1 channel, 4097 samples)
    # Using random data as a placeholder. In a real parity test, 
    # you'd load a known sample from the notebook that produced a specific prediction.
    eeg_data = np.random.rand(1, 4097)
    channel_names = ["Z"]
    
    result = predictor.get_prediction(eeg_data, channel_names, 173.61, "lightgbm")
    
    assert "prediction" in result
    assert "probabilities" in result["prediction"]
    assert "explanation" in result
    assert result["prediction"]["label"] in ["seizure", "non_seizure"]

def test_chbmit_inference(load_models):
    try:
        predictor = load_models.get_predictor("chbmit")
    except ValueError:
        pytest.skip("CHB-MIT model not loaded or not available")
        
    # Mock CHB-MIT dataset (23 channels, 15360 samples)
    eeg_data = np.random.rand(23, 15360)
    channel_names = [f"Ch{i}" for i in range(23)]
    
    result = predictor.get_prediction(eeg_data, channel_names, 256.0, "lightgbm")
    
    assert "prediction" in result
    assert "probabilities" in result["prediction"]
    assert "explanation" in result
    assert result["prediction"]["label"] in ["seizure", "non_seizure"]
