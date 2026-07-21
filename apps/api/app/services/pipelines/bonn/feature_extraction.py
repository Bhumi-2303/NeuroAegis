import numpy as np
import pywt
from scipy.stats import skew, kurtosis, entropy, iqr
from scipy.signal import welch

# Constants strictly migrated from notebook
FS = 173.61

def hjorth_parameters(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    activity = np.var(signal)

    mobility = np.sqrt(
        np.var(first_deriv) /
        (activity + 1e-12)
    )

    complexity = np.sqrt(
        np.var(second_deriv) /
        (np.var(first_deriv) + 1e-12)
    ) / (mobility + 1e-12)

    return activity, mobility, complexity

def zero_crossing_rate(signal):
    return np.sum(np.diff(np.sign(signal)) != 0)

def line_length(signal):
    return np.sum(np.abs(np.diff(signal)))

def bandpower(freqs, psd, low, high):
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx], freqs[idx])

def frequency_features(signal):
    freqs, psd = welch(
        signal,
        fs=FS,
        nperseg=512
    )

    total_power = np.trapz(psd, freqs) + 1e-12

    delta = bandpower(freqs, psd, 0.5, 4)
    theta = bandpower(freqs, psd, 4, 8)
    alpha = bandpower(freqs, psd, 8, 13)
    beta = bandpower(freqs, psd, 13, 30)
    gamma = bandpower(freqs, psd, 30, 45)

    dominant_frequency = freqs[np.argmax(psd)]
    spectral_entropy = entropy(psd / np.sum(psd))
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)

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
        "spectral_entropy": spectral_entropy,
        "spectral_centroid": spectral_centroid,
        "total_power": total_power
    }

def wavelet_features(signal):
    coeffs = pywt.wavedec(
        signal,
        "coif3",
        level=4
    )

    energies = [
        np.sum(c**2)
        for c in coeffs
    ]

    total_energy = np.sum(energies) + 1e-12
    probs = np.array(energies) / total_energy

    features = {}
    wavelet_entropy = entropy(probs)
    features["wavelet_entropy"] = wavelet_entropy

    for i, c in enumerate(coeffs):
        features[f"wavelet_energy_{i}"] = energies[i]
        features[f"wavelet_relative_energy_{i}"] = (energies[i] / total_energy)
        features[f"wavelet_mean_{i}"] = np.mean(c)
        features[f"wavelet_std_{i}"] = np.std(c)

    return features

def time_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    abs_mean = np.mean(np.abs(signal))
    peak = np.max(np.abs(signal))
    activity, mobility, complexity = hjorth_parameters(signal)

    crest_factor = peak / (rms + 1e-12)
    shape_factor = rms / (abs_mean + 1e-12)
    impulse_factor = peak / (abs_mean + 1e-12)
    clearance_factor = peak / ((np.mean(np.sqrt(np.abs(signal)))**2) + 1e-12)

    return {
        "mean": np.mean(signal),
        "median": np.median(signal),
        "std": np.std(signal),
        "variance": np.var(signal),
        "minimum": np.min(signal),
        "maximum": np.max(signal),
        "range": np.ptp(signal),
        "rms": rms,
        "energy": np.sum(signal**2),
        "absolute_mean": abs_mean,
        "line_length": line_length(signal),
        "zero_crossings": zero_crossing_rate(signal),
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal),
        "iqr": iqr(signal),
        "crest_factor": crest_factor,
        "shape_factor": shape_factor,
        "impulse_factor": impulse_factor,
        "clearance_factor": clearance_factor,
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity
    }

def extract_features(signal):
    features = {}
    features.update(time_features(signal))
    features.update(frequency_features(signal))
    features.update(wavelet_features(signal))
    return features

def extract_all_features(data: np.ndarray, channel_names: list = None, fs: float = 173.61) -> dict:
    """Wrapper to integrate with existing API flow"""
    if data.ndim > 1:
        data = data.flatten()
    return extract_features(data)
