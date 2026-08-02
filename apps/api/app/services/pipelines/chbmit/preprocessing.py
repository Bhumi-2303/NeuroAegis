"""CHB-MIT dataset preprocessing — thin wrapper around the shared features module."""

from typing import List, Union

import numpy as np
import pandas as pd

from app.services.features import (
    preprocess_eeg as _preprocess_eeg,
    wavelet_denoise,  # re-export for backwards compatibility
)

# CHB-MIT-specific defaults: db4 wavelet, 5-level decomposition
_CHBMIT_WAVELET = "db4"
_CHBMIT_LEVEL = 5


def preprocess_eeg(data: Union[np.ndarray, List[float], pd.DataFrame]) -> np.ndarray:
    """Preprocess EEG data for CHB-MIT.

    1. Convert to numpy array
    2. Wavelet Denoising (db4, level 5 for CHB-MIT)
    """
    return _preprocess_eeg(data, wavelet=_CHBMIT_WAVELET, level=_CHBMIT_LEVEL)
