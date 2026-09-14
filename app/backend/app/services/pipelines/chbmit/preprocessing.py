from __future__ import annotations
"""CHB-MIT dataset preprocessing — thin wrapper around the shared features module."""


import numpy as np
import pandas as pd

from app.services.features import (
    preprocess_eeg as _preprocess_eeg,
)

# CHB-MIT-specific defaults: db4 wavelet, 5-level decomposition
_CHBMIT_WAVELET = "db4"
_CHBMIT_LEVEL = 5


def preprocess_eeg(data: np.ndarray | list[float] | pd.DataFrame) -> np.ndarray:
    """Preprocess EEG data for CHB-MIT.

    1. Convert to numpy array
    2. Wavelet Denoising (db4, level 5 for CHB-MIT)
    """
    return _preprocess_eeg(data, wavelet=_CHBMIT_WAVELET, level=_CHBMIT_LEVEL)
