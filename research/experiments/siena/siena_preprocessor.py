"""
siena_preprocessor.py
──────────────────────
NeuroAegis Research — Phase 6: Cross-Dataset Generalization (Siena)
Harmonizes Siena Scalp EEG (512 Hz, referential montage) into the exact
input contract expected by the frozen Phase 4B CNN_GNN_GRU model:
  1. 23 Canonical Bipolar Channels (derived from 29 referential electrodes)
  2. 256 Hz Sampling Frequency (decimated by factor of 2 with anti-aliasing)
  3. Zero-phase Bandpass Filter: 0.5 - 40.0 Hz (Butterworth Order 4 SOS)
  4. Zero-phase Notch Filter: 50.0 Hz (Q=30.0, European powerline)
  5. Recording-local Z-score Normalization ((x - mean) / std)
  6. 5.0s Windowing (1280 samples) with 2.5s stride (640 samples)
  7. Causal GRU Sequence Construction (L=8, 22.5s temporal span)
"""

import os
import json
import numpy as np
import scipy.signal
import mne
from typing import List, Tuple, Dict, Optional

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
CONFIG_DIR = os.path.join(BASE_DIR, "research/experiments/siena/config")
CHANNEL_MAPPING_PATH = os.path.join(CONFIG_DIR, "siena_channel_mapping.json")


class SienaChannelHarmonizer:
    """
    Reconstructs the 23 canonical CHB-MIT bipolar channels from Siena referential channels.
    """
    def __init__(self, mapping_path: str = CHANNEL_MAPPING_PATH):
        with open(mapping_path, "r") as f:
            self.mapping_config = json.load(f)
        self.mappings = self.mapping_config["mappings"]
        self.bipolar_names = [m["source_channel"] for m in self.mappings]
        self.derivation_pairs = [(m["target_anode"], m["target_cathode"]) for m in self.mappings]
        
    def harmonize_channels(self, raw_data: np.ndarray, channel_names: List[str]) -> np.ndarray:
        """
        Derives 23 bipolar channels from raw referential array (n_channels, n_samples).
        """
        # Clean channel names: strip 'EEG ' and whitespace, uppercase
        clean_names = [c.replace("EEG ", "").strip().upper() for c in channel_names]
        name_to_idx = {name: i for i, name in enumerate(clean_names)}
        
        n_samples = raw_data.shape[1]
        bipolar_data = np.zeros((len(self.derivation_pairs), n_samples), dtype=np.float32)
        
        for i, (anode, cathode) in enumerate(self.derivation_pairs):
            if anode not in name_to_idx:
                raise ValueError(f"Required anode channel '{anode}' missing from recording!")
            if cathode not in name_to_idx:
                raise ValueError(f"Required cathode channel '{cathode}' missing from recording!")
            bipolar_data[i] = raw_data[name_to_idx[anode]] - raw_data[name_to_idx[cathode]]
            
        return bipolar_data


class SienaPreprocessor:
    """
    Signal filtering and normalization pipeline.
    Matches frozen Phase 2 / Phase 4B specifications.
    """
    def __init__(
        self,
        orig_sfreq: float = 512.0,
        target_sfreq: float = 256.0,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        notch_freq: float = 50.0,
        notch_q: float = 30.0,
        filter_order: int = 4,
        epsilon: float = 1e-8
    ):
        self.orig_sfreq = orig_sfreq
        self.target_sfreq = target_sfreq
        self.decimation_factor = int(orig_sfreq // target_sfreq)
        self.epsilon = epsilon
        
        # Design zero-phase bandpass filter at target sfreq (256 Hz)
        self.sos_bp = scipy.signal.butter(
            filter_order,
            [lowcut, highcut],
            btype="bandpass",
            fs=target_sfreq,
            output="sos"
        )
        # Design notch filter at 50.0 Hz (European powerline)
        self.b_notch, self.a_notch = scipy.signal.iirnotch(notch_freq, notch_q, fs=target_sfreq)
        
    def process_bipolar_data(self, bipolar_data: np.ndarray) -> np.ndarray:
        """
        1. Decimate 512 Hz -> 256 Hz using zero-phase anti-aliasing Chebyshev decimation.
        2. Bandpass filter 0.5 - 40.0 Hz (forward-backward SOS).
        3. Notch filter 50.0 Hz (forward-backward IIR).
        4. Recording-local z-score normalization.
        """
        # 1. Decimate along time axis
        resampled = scipy.signal.decimate(bipolar_data, q=self.decimation_factor, axis=-1, zero_phase=True)
        
        # 2. Bandpass SOS
        filtered = scipy.signal.sosfiltfilt(self.sos_bp, resampled, axis=-1)
        
        # 3. Notch IIR
        filtered = scipy.signal.filtfilt(self.b_notch, self.a_notch, filtered, axis=-1)
        
        # 4. Recording-local z-score
        mean = np.mean(filtered, axis=-1, keepdims=True)
        std = np.std(filtered, axis=-1, keepdims=True)
        std = np.where(std < self.epsilon, 1.0, std)
        normalized = (filtered - mean) / std
        
        return normalized.astype(np.float32)


class SienaWindowExtractor:
    """
    Extracts 5.0s windows (1280 samples) with 2.5s stride (640 samples).
    """
    def __init__(self, window_samples: int = 1280, stride_samples: int = 640):
        self.window_samples = window_samples
        self.stride_samples = stride_samples
        
    def extract_windows(self, normalized_data: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Returns:
            windows: Tensor of shape (n_windows, 23, 1280)
            time_intervals: List of (start_sec, end_sec) for each window
        """
        n_channels, total_samples = normalized_data.shape
        n_windows = (total_samples - self.window_samples) // self.stride_samples + 1
        
        windows = np.zeros((n_windows, n_channels, self.window_samples), dtype=np.float32)
        intervals = []
        
        for w in range(n_windows):
            start = w * self.stride_samples
            end = start + self.window_samples
            windows[w] = normalized_data[:, start:end]
            intervals.append((start / 256.0, end / 256.0))
            
        return windows, intervals


class SienaSequenceBuilder:
    """
    Constructs causal GRU sequences of length L=8 (22.5s span) from window arrays.
    """
    def __init__(self, seq_len: int = 8):
        self.seq_len = seq_len
        
    def build_causal_sequences(self, windows: np.ndarray) -> np.ndarray:
        """
        windows: (n_windows, 23, 1280)
        Returns: (n_windows, seq_len, 23, 1280) with causal zero-padding for w < seq_len - 1.
        """
        n_windows, n_channels, n_samples = windows.shape
        sequences = np.zeros((n_windows, self.seq_len, n_channels, n_samples), dtype=np.float32)
        
        for w in range(n_windows):
            start_hist = max(0, w - self.seq_len + 1)
            hist_len = w - start_hist + 1
            sequences[w, self.seq_len - hist_len :] = windows[start_hist : w + 1]
            
        return sequences
