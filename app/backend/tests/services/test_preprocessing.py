import numpy as np
import pytest

from app.services.pipelines.bonn.preprocessing import preprocess_eeg as bonn_preprocess
from app.services.pipelines.chbmit.preprocessing import (
    preprocess_eeg as chbmit_preprocess,
)


@pytest.fixture
def sample_signal():
    np.random.seed(42)
    return np.random.randn(4097)

def test_bonn_preprocessing(sample_signal):
    # Test that bonn preprocessing preserves length and returns ndarray
    result = bonn_preprocess(sample_signal)
    assert len(result) == len(sample_signal)
    assert isinstance(result, np.ndarray)
    assert not np.array_equal(result, sample_signal)

def test_chbmit_preprocessing(sample_signal):
    # Test that chbmit preprocessing preserves length and returns ndarray
    result = chbmit_preprocess(sample_signal)
    assert len(result) == len(sample_signal)
    assert isinstance(result, np.ndarray)
    assert not np.array_equal(result, sample_signal)

def test_preprocessing_differences(sample_signal):
    # Bonn uses coif3/level4, CHB-MIT uses db4/level5. 
    # They should produce different outputs for the same signal.
    bonn_result = bonn_preprocess(sample_signal)
    chbmit_result = chbmit_preprocess(sample_signal)
    assert not np.array_equal(bonn_result, chbmit_result)
