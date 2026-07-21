import pytest
import pandas as pd
import numpy as np
from app.services.dataset_detection.detector import DatasetDetector
from app.services.dataset_detection.metadata import DatasetMetadata
from app.services.dataset_detection.rules import DefaultDatasetScorer

@pytest.fixture
def mock_datasets():
    return {
        "bonn": DatasetMetadata(
            id="bonn",
            enabled=True,
            path="",
            default_model="lightgbm",
            display_name="Bonn",
            expected_channels=1,
            sampling_rate=173.61,
            window_length=4097,
            feature_count=5,
            supported_extensions=[".csv"]
        ),
        "chbmit": DatasetMetadata(
            id="chbmit",
            enabled=True,
            path="",
            default_model="lightgbm",
            display_name="CHB-MIT",
            expected_channels=23,
            sampling_rate=256.0,
            window_length=15360,
            feature_count=36864,
            supported_extensions=[".csv"]
        )
    }

def test_bonn_detection(mock_datasets):
    detector = DatasetDetector()
    detector.datasets = mock_datasets
    
    # Create fake Bonn data (1 channel, 4097 samples)
    data = np.random.rand(4097, 1)
    df = pd.DataFrame(data, columns=["Z"])
    
    dataset_id, confidence, rules = detector.detect(df, provided_fs=173.61)
    assert dataset_id == "bonn"
    assert confidence > 0.90

def test_chbmit_detection(mock_datasets):
    detector = DatasetDetector()
    detector.datasets = mock_datasets
    
    # Create fake CHB-MIT data (23 channels, 15360 samples)
    data = np.random.rand(15360, 23)
    cols = [f"Ch{i}" for i in range(23)]
    df = pd.DataFrame(data, columns=cols)
    
    dataset_id, confidence, rules = detector.detect(df, provided_fs=256.0)
    assert dataset_id == "chbmit"
    assert confidence > 0.90

def test_invalid_csv():
    detector = DatasetDetector()
    # Empty DF
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="CSV file is empty"):
        detector.detect(df, provided_fs=256.0)
        
    # Non-numeric DF
    df2 = pd.DataFrame({"A": ["a", "b"], "B": [1, 2]})
    with pytest.raises(ValueError, match="numeric EEG data"):
        detector.detect(df2, provided_fs=256.0)
