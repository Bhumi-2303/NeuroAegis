"""
NeuroAegis — Shared EEG Feature Extraction, Wavelet Denoising & Preprocessing
==============================================================================

Canonical, unit-testable implementations extracted from neuroaegis-v1.ipynb
(upgraded feature library, Cell 12).  Every signal-processing function used
by the Bonn and CHB-MIT prediction pipelines lives here; dataset-specific
pipeline modules are thin wrappers that call into this module with the
appropriate default parameters.

Public API
----------
Denoising & preprocessing:
    wavelet_denoise, preprocess_eeg

Feature helpers:
    hjorth_parameters, zero_crossing_rate, line_length, bandpower

Feature groups:
    time_features, frequency_features, wavelet_features

Master extractors:
    extract_features, extract_features_multichannel

Feature selection:
    select_and_order_features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
from scipy.signal import welch
from scipy.stats import entropy, iqr, kurtosis, skew

# ── Constants ────────────────────────────────────────────────────────
DEFAULT_FS: float = 173.61
"""Default sampling frequency (Hz) — Bonn University dataset."""

DEFAULT_WAVELET: str = "coif3"
"""Default wavelet family for DWT decomposition."""

DEFAULT_LEVEL: int = 4
"""Default decomposition level for DWT."""


# =====================================================================
# Denoising & Preprocessing
# =====================================================================

def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> np.ndarray:
    """Apply Discrete Wavelet Transform (DWT) denoising with soft thresholding.

    Uses the universal threshold rule (VisuShrink):
        threshold = σ · √(2 · ln(N))
    where σ is estimated from the finest-scale detail coefficients via the
    Median Absolute Deviation (MAD).

    Parameters
    ----------
    signal : np.ndarray
        1-D array of EEG samples.
    wavelet : str
        Wavelet family identifier (e.g. ``"coif3"``, ``"db4"``).
    level : int
        Number of decomposition levels.

    Returns
    -------
    np.ndarray
        Denoised signal, same length as *signal*.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Noise variance estimate via MAD of finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Preserve approximation coefficients; soft-threshold details
    coeffs_denoised = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_denoised.append(pywt.threshold(c, threshold, mode="soft"))

    reconstructed = pywt.waverec(coeffs_denoised, wavelet)
    return reconstructed[: len(signal)]


def preprocess_eeg(
    data: np.ndarray | list[float] | pd.DataFrame,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> np.ndarray:
    """Preprocess raw EEG data: coerce to 1-D array and apply wavelet denoising.

    Parameters
    ----------
    data : array-like or DataFrame
        Raw EEG recording.
    wavelet : str
        Wavelet family for denoising (default ``"coif3"`` for Bonn).
    level : int
        DWT decomposition depth.

    Returns
    -------
    np.ndarray
        1-D denoised signal.
    """
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.DataFrame):
        data = data.values

    if data.ndim > 1:
        data = data.flatten()

    return wavelet_denoise(data, wavelet=wavelet, level=level)


# =====================================================================
# Feature Helpers
# =====================================================================

def hjorth_parameters(signal: np.ndarray) -> tuple[float, float, float]:
    """Compute Hjorth activity, mobility, and complexity.

    Parameters
    ----------
    signal : np.ndarray
        1-D EEG segment.

    Returns
    -------
    tuple of float
        ``(activity, mobility, complexity)``
    """
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    activity = float(np.var(signal))
    mobility = float(np.sqrt(np.var(first_deriv) / (activity + 1e-12)))
    complexity = float(
        np.sqrt(np.var(second_deriv) / (np.var(first_deriv) + 1e-12))
        / (mobility + 1e-12)
    )
    return activity, mobility, complexity


def zero_crossing_rate(signal: np.ndarray) -> int:
    """Count the number of zero crossings in *signal*.

    Parameters
    ----------
    signal : np.ndarray
        1-D EEG segment.

    Returns
    -------
    int
        Number of sign changes.
    """
    return int(np.sum(np.diff(np.sign(signal)) != 0))


def line_length(signal: np.ndarray) -> float:
    """Compute signal line length (sum of absolute first differences).

    Parameters
    ----------
    signal : np.ndarray
        1-D EEG segment.

    Returns
    -------
    float
    """
    return float(np.sum(np.abs(np.diff(signal))))


def bandpower(
    freqs: np.ndarray,
    psd: np.ndarray,
    low: float,
    high: float,
) -> float:
    """Integrate PSD over a frequency band using the trapezoidal rule.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency bins from Welch's method.
    psd : np.ndarray
        Power spectral density values.
    low, high : float
        Band edges in Hz.

    Returns
    -------
    float
        Band power.
    """
    idx = np.logical_and(freqs >= low, freqs <= high)
    return float(np.trapz(psd[idx], freqs[idx]))


# =====================================================================
# Feature Groups
# =====================================================================

def time_features(signal: np.ndarray) -> dict[str, float]:
    """Extract 22 time-domain features from an EEG segment.

    Includes statistical moments, Hjorth parameters, shape factors, and
    waveform descriptors.

    Parameters
    ----------
    signal : np.ndarray
        1-D EEG segment.

    Returns
    -------
    dict
        Feature name → value mapping (22 entries).
    """
    rms = float(np.sqrt(np.mean(signal ** 2)))
    abs_mean = float(np.mean(np.abs(signal)))
    peak = float(np.max(np.abs(signal)))
    activity, mobility, complexity = hjorth_parameters(signal)

    crest_factor = peak / (rms + 1e-12)
    shape_factor = rms / (abs_mean + 1e-12)
    impulse_factor = peak / (abs_mean + 1e-12)
    clearance_factor = peak / ((np.mean(np.sqrt(np.abs(signal))) ** 2) + 1e-12)

    return {
        "mean": float(np.mean(signal)),
        "median": float(np.median(signal)),
        "std": float(np.std(signal)),
        "variance": float(np.var(signal)),
        "minimum": float(np.min(signal)),
        "maximum": float(np.max(signal)),
        "range": float(np.ptp(signal)),
        "rms": rms,
        "energy": float(np.sum(signal ** 2)),
        "absolute_mean": abs_mean,
        "line_length": line_length(signal),
        "zero_crossings": zero_crossing_rate(signal),
        "skewness": float(skew(signal)),
        "kurtosis": float(kurtosis(signal)),
        "iqr": float(iqr(signal)),
        "crest_factor": crest_factor,
        "shape_factor": shape_factor,
        "impulse_factor": impulse_factor,
        "clearance_factor": clearance_factor,
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def frequency_features(
    signal: np.ndarray,
    fs: float = DEFAULT_FS,
) -> dict[str, float]:
    """Extract 14 frequency-domain features from an EEG segment.

    Uses Welch's method for PSD estimation.  Computes absolute and relative
    band powers for δ, θ, α, β, γ bands, plus spectral centroid, entropy,
    dominant frequency, and total power.

    Parameters
    ----------
    signal : np.ndarray
        1-D EEG segment.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Feature name → value mapping (14 entries).
    """
    freqs, psd = welch(signal, fs=fs, nperseg=min(512, len(signal)))

    total_power = float(np.trapz(psd, freqs)) + 1e-12

    delta = bandpower(freqs, psd, 0.5, 4)
    theta = bandpower(freqs, psd, 4, 8)
    alpha = bandpower(freqs, psd, 8, 13)
    beta = bandpower(freqs, psd, 13, 30)
    gamma = bandpower(freqs, psd, 30, 45)

    dominant_frequency = float(freqs[np.argmax(psd)])
    spectral_entropy_val = float(entropy(psd / np.sum(psd)))
    spectral_centroid = float(np.sum(freqs * psd) / np.sum(psd))

    return {
        "delta_power": delta,
        "theta_power": theta,
        "alpha_power": alpha,
        "beta_power": beta,
        "gamma_power": gamma,
        "relative_delta": delta / total_power,
        "relative_theta": theta / total_power,
        "relative_alpha": alpha / total_power,
        "relative_beta": beta / total_power,
        "relative_gamma": gamma / total_power,
        "dominant_frequency": dominant_frequency,
        "spectral_entropy": spectral_entropy_val,
        "spectral_centroid": spectral_centroid,
        "total_power": total_power,
    }


def wavelet_features(
    signal: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> dict[str, float]:
    """Extract 21 wavelet-domain features from an EEG segment.

    Decomposes *signal* with a DWT and computes per-level energy, relative
    energy, mean, and std of wavelet coefficients, plus an overall wavelet
    entropy.

    Parameters
    ----------
    signal : np.ndarray
        1-D EEG segment.
    wavelet : str
        Wavelet family identifier.
    level : int
        Number of decomposition levels.

    Returns
    -------
    dict
        Feature name → value mapping (21 entries for 4-level decomposition).
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    energies = [float(np.sum(c ** 2)) for c in coeffs]
    total_energy = sum(energies) + 1e-12
    probs = np.array(energies) / total_energy

    features: dict[str, float] = {}
    features["wavelet_entropy"] = float(entropy(probs))

    for i, c in enumerate(coeffs):
        features[f"wavelet_energy_{i}"] = energies[i]
        features[f"wavelet_relative_energy_{i}"] = energies[i] / total_energy
        features[f"wavelet_mean_{i}"] = float(np.mean(c))
        features[f"wavelet_std_{i}"] = float(np.std(c))

    return features


# =====================================================================
# Master Extractors
# =====================================================================

def extract_features(
    signal: np.ndarray,
    fs: float = DEFAULT_FS,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> dict[str, float]:
    """Extract the full 57-feature vector from a single EEG segment.

    Combines time-domain (22), frequency-domain (14), and wavelet-domain
    (21) features into a single dictionary.

    Parameters
    ----------
    signal : np.ndarray
        1-D EEG segment.
    fs : float
        Sampling frequency in Hz.
    wavelet : str
        Wavelet family for wavelet features.
    level : int
        DWT decomposition depth.

    Returns
    -------
    dict
        Feature name → value mapping (57 entries).
    """
    features: dict[str, float] = {}
    features.update(time_features(signal))
    features.update(frequency_features(signal, fs=fs))
    features.update(wavelet_features(signal, wavelet=wavelet, level=level))
    return features


def extract_features_multichannel(
    data: np.ndarray,
    channel_names: list[str] | None = None,
    fs: float = DEFAULT_FS,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> dict[str, float]:
    """Extract features from multi-channel EEG data.

    For each channel, extracts the full feature vector and prefixes each
    feature name with the channel name / index.  If *data* is 1-D it is
    treated as a single channel.

    Parameters
    ----------
    data : np.ndarray
        EEG recording — shape ``(n_channels, n_samples)`` or 1-D.
    channel_names : list of str, optional
        Human-readable channel labels.  Defaults to ``Ch0, Ch1, …``
    fs : float
        Sampling frequency in Hz.
    wavelet : str
        Wavelet family for wavelet features.
    level : int
        DWT decomposition depth.

    Returns
    -------
    dict
        Feature name → value mapping (57 × n_channels entries).
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_channels = data.shape[0]
    if channel_names is None or len(channel_names) != n_channels:
        channel_names = [f"Ch{i}" for i in range(n_channels)]

    all_features: dict[str, float] = {}
    for ch_idx, ch_name in enumerate(channel_names):
        ch_features = extract_features(
            data[ch_idx], fs=fs, wavelet=wavelet, level=level
        )
        for feat_name, feat_val in ch_features.items():
            all_features[f"{ch_name}_{feat_name}"] = feat_val

    return all_features


# =====================================================================
# Feature Selection
# =====================================================================

def select_and_order_features(
    extracted_features: dict[str, float],
    selected_features: list[str],
) -> np.ndarray:
    """Select and order features to match the trained model's expectations.

    Parameters
    ----------
    extracted_features : dict
        Full feature dictionary (from ``extract_features`` or
        ``extract_features_multichannel``).
    selected_features : list of str
        Ordered feature names matching the model's training schema.

    Returns
    -------
    np.ndarray
        Feature vector of shape ``(1, n_features)`` ready for prediction.

    Raises
    ------
    ValueError
        If any required feature is missing from *extracted_features*.
    """
    extracted_keys = set(extracted_features.keys())
    selected_keys = set(selected_features)

    missing = selected_keys - extracted_keys
    if missing:
        raise ValueError(
            f"Feature Validation Error: Missing {len(missing)} "
            f"required features: {missing}"
        )

    feature_vector = [extracted_features[name] for name in selected_features]
    return np.array(feature_vector).reshape(1, -1)
