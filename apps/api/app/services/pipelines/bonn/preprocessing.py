"""Bonn dataset preprocessing — thin wrapper around the shared features module."""

from typing import List, Union

import numpy as np
import pandas as pd

from app.services.features import (
    preprocess_eeg as _preprocess_eeg,
    wavelet_denoise,  # re-export for backwards compatibility
)

# Bonn-specific defaults: coif3 wavelet, 4-level decomposition
_BONN_WAVELET = "coif3"
_BONN_LEVEL = 4


def preprocess_eeg(data: Union[np.ndarray, List[float], pd.DataFrame]) -> np.ndarray:
    """Preprocess EEG data using the Bonn dataset methodology.

    1. Convert to numpy array
    2. Wavelet Denoising (coif3, level 4)
    """
    return _preprocess_eeg(data, wavelet=_BONN_WAVELET, level=_BONN_LEVEL)
