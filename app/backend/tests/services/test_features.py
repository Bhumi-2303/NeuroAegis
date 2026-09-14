import numpy as np
import pytest

from app.services.features import (
    bandpower,
    extract_features,
    extract_features_multichannel,
    frequency_features,
    hjorth_parameters,
    line_length,
    preprocess_eeg,
    select_and_order_features,
    time_features,
    wavelet_denoise,
    wavelet_features,
    zero_crossing_rate,
)


@pytest.fixture
def sample_signal():
    np.random.seed(42)
    return np.random.randn(4097)

def test_wavelet_denoise(sample_signal):
    denoised = wavelet_denoise(sample_signal, wavelet='coif3', level=4)
    assert len(denoised) == len(sample_signal)
    assert not np.array_equal(denoised, sample_signal)
    assert isinstance(denoised, np.ndarray)

def test_preprocess_eeg(sample_signal):
    # Test with numpy array
    preprocessed = preprocess_eeg(sample_signal)
    assert len(preprocessed) == len(sample_signal)
    
    # Test with list
    preprocessed_list = preprocess_eeg(sample_signal.tolist())
    assert len(preprocessed_list) == len(sample_signal)

def test_hjorth_parameters(sample_signal):
    activity, mobility, complexity = hjorth_parameters(sample_signal)
    assert isinstance(activity, float)
    assert isinstance(mobility, float)
    assert isinstance(complexity, float)

def test_zero_crossing_rate():
    signal = np.array([1, -1, 1, -1, 1])
    assert zero_crossing_rate(signal) == 4

def test_line_length():
    signal = np.array([0, 1, 0, 1, 0])
    assert line_length(signal) == 4.0

def test_bandpower():
    freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    psd = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    bp = bandpower(freqs, psd, 2.0, 4.0)
    # Area under y=1 from x=2 to x=4 is 2.0
    assert bp == 2.0

def test_time_features(sample_signal):
    features = time_features(sample_signal)
    assert len(features) == 22
    assert "mean" in features
    assert "kurtosis" in features
    assert "hjorth_complexity" in features

def test_frequency_features(sample_signal):
    features = frequency_features(sample_signal, fs=173.61)
    assert len(features) == 14
    assert "delta_power" in features
    assert "spectral_entropy" in features

def test_wavelet_features(sample_signal):
    features_lvl4 = wavelet_features(sample_signal, wavelet='coif3', level=4)
    # level 4 -> 5 coefficient arrays -> 1 entropy + 5*4 = 21 features
    assert len(features_lvl4) == 21
    
    features_lvl5 = wavelet_features(sample_signal, wavelet='db4', level=5)
    # level 5 -> 6 coefficient arrays -> 1 entropy + 6*4 = 25 features
    assert len(features_lvl5) == 25

def test_extract_features(sample_signal):
    # Bonn params
    features_bonn = extract_features(sample_signal, fs=173.61, wavelet='coif3', level=4)
    assert len(features_bonn) == 57
    
    # CHBMIT params
    features_chb = extract_features(sample_signal, fs=256.0, wavelet='db4', level=5)
    assert len(features_chb) == 61

def test_extract_features_multichannel():
    np.random.seed(42)
    data = np.random.randn(2, 4097)
    features = extract_features_multichannel(data, channel_names=["Ch1", "Ch2"], fs=256.0, wavelet='db4', level=5)
    assert len(features) == 61 * 2
    assert "Ch1_mean" in features
    assert "Ch2_mean" in features

def test_select_and_order_features(sample_signal):
    features = extract_features(sample_signal, fs=173.61, wavelet='coif3', level=4)
    selected_keys = ["mean", "std", "kurtosis"]
    ordered = select_and_order_features(features, selected_keys)
    assert ordered.shape == (1, 3)
    assert ordered[0][0] == features["mean"]
    
    # Missing feature should raise ValueError
    with pytest.raises(ValueError):
        select_and_order_features(features, ["mean", "does_not_exist"])
