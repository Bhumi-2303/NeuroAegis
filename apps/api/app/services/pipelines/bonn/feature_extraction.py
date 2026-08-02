"""Bonn dataset feature extraction — thin wrapper around the shared features module."""

import numpy as np

from app.services.features import (
    extract_features,
    hjorth_parameters,
    zero_crossing_rate,
    line_length,
    bandpower,
    frequency_features,
    wavelet_features,
    time_features,
)

# Bonn-specific constants
FS = 173.61


def extract_all_features(
    data: np.ndarray,
    channel_names: list = None,
    fs: float = FS,
) -> dict:
    """Extract all features from Bonn EEG data.

    Wrapper to integrate with existing API flow.  Flattens multi-dimensional
    input and delegates to the shared ``extract_features`` function.
    """
    if data.ndim > 1:
        data = data.flatten()
    return extract_features(data, fs=fs)
