import numpy as np
import pywt
from typing import Union, List
import pandas as pd

def wavelet_denoise(signal: np.ndarray, wavelet: str = "db4", level: int = 5) -> np.ndarray:
    """
    Apply Discrete Wavelet Transform (DWT) for denoising.
    Dedicated production module for CHB-MIT.
    """
    coeffs = pywt.wavedec(
        signal,
        wavelet,
        level=level
    )

    sigma = np.median(
        np.abs(coeffs[-1])
    ) / 0.6745

    threshold = sigma * np.sqrt(
        2 * np.log(len(signal))
    )

    coeffs_denoised = [coeffs[0]]

    for c in coeffs[1:]:
        coeffs_denoised.append(
            pywt.threshold(
                c,
                threshold,
                mode="soft"
            )
        )

    reconstructed = pywt.waverec(
        coeffs_denoised,
        wavelet
    )

    return reconstructed[:len(signal)]

def preprocess_eeg(data: Union[np.ndarray, List[float], pd.DataFrame]) -> np.ndarray:
    """
    Preprocess the EEG data for CHB-MIT.
    1. Convert to numpy array
    2. Wavelet Denoising (db4, level 5 for CHB-MIT)
    """
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.DataFrame):
        data = data.values
        
    if data.ndim > 1:
        # Assuming single channel or flattened data based on previous setup
        data = data.flatten()
        
    # Apply wavelet denoising
    denoised_data = wavelet_denoise(data, wavelet="db4", level=5)
    return denoised_data
