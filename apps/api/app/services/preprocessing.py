import numpy as np
import pywt
from typing import Union, List
import pandas as pd

def wavelet_denoise(signal: np.ndarray, wavelet: str = "coif3", level: int = 4) -> np.ndarray:
    """
    Apply Discrete Wavelet Transform (DWT) for denoising.
    Directly migrated from neuroaegis-v1.ipynb.
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
    Preprocess the EEG data using the exact methodology from training:
    1. Convert to numpy array
    2. Wavelet Denoising
    """
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.DataFrame):
        data = data.values
        
    if data.ndim > 1:
        # Assuming single channel for now based on the notebook logic
        data = data.flatten()
        
    # Apply wavelet denoising
    denoised_data = wavelet_denoise(data, wavelet="coif3", level=4)
    return denoised_data
