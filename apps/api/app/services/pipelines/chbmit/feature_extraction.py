"""CHB-MIT dataset feature extraction — delegates to the shared features module."""

import numpy as np
from typing import Dict, List

from app.services.features import extract_features, extract_features_multichannel

# CHB-MIT-specific constants
FS = 256.0


def extract_all_features(
    signal: np.ndarray,
    channel_names: List[str],
    fs: float = FS,
) -> Dict[str, float]:
    """Extract features from preprocessed CHB-MIT EEG signal.

    For multi-channel data (2-D array with shape ``(n_channels, n_samples)``),
    extracts per-channel features and concatenates them.  For 1-D data,
    extracts a single set of features.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed EEG data — 1-D or 2-D ``(n_channels, n_samples)``.
    channel_names : list of str
        Channel labels.
    fs : float
        Sampling frequency in Hz (default 256.0 for CHB-MIT).

    Returns
    -------
    dict
        Feature name → value mapping.
    """
    if signal.ndim > 1:
        return extract_features_multichannel(
            signal, channel_names=channel_names, fs=fs, wavelet="db4", level=5,
        )

    return extract_features(signal, fs=fs, wavelet="db4", level=5)
