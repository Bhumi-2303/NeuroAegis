"""
NeuroAegis CHB-MIT EEG Preprocessor & Streaming Pipeline
Phase 2: Signal Filtering, Montage Standardization, Leakage-Safe Normalization, and Window Access
"""

import os
import json
import numpy as np
import scipy.signal as signal
import mne

class CHBMITChannelManager:
    """Manages canonical 23-channel bipolar montage extraction and mapping."""
    
    def __init__(self, channel_order_path=None):
        if channel_order_path is None:
            channel_order_path = "research/data/config/chbmit_channel_order.json"
        with open(channel_order_path, "r") as f:
            self.canonical_channels = json.load(f)
        self.expected_count = len(self.canonical_channels)  # 23
        
    def get_canonical_channels(self):
        return list(self.canonical_channels)
        
    def map_recording_channels(self, edf_raw_ch_names):
        """
        Maps raw EDF channel names to canonical 23 channels in exact fixed order.
        Returns (picked_indices, montage_status, missing_channels)
        """
        clean_names = [ch.strip().upper().replace(".", "") for ch in edf_raw_ch_names]
        
        picked_indices = []
        missing_channels = []
        
        for i, target_ch in enumerate(self.canonical_channels):
            matches = [idx for idx, name in enumerate(clean_names) if name == target_ch or name.startswith(target_ch + "-")]
            if len(matches) == 1:
                picked_indices.append(matches[0])
            elif len(matches) > 1:
                # Target T8-P8 appears at index 14 and index 22
                if i == 22 and len(matches) >= 2:
                    picked_indices.append(matches[1])
                else:
                    picked_indices.append(matches[0])
            else:
                missing_channels.append(target_ch)
                
        if len(picked_indices) == self.expected_count and len(missing_channels) == 0:
            montage_status = "CANONICAL_23"
        elif any("-CS2" in name for name in clean_names):
            montage_status = "COMMON_REF_CS2"
        else:
            montage_status = f"MODIFIED_{len(picked_indices)}_CHANNELS"
            
        return picked_indices, montage_status, missing_channels


class CHBMITSignalFilter:
    """
    Applies configurable zero-phase bandpass and notch filtering.
    Zero-phase filtering is implemented via forward-backward second-order sections (sosfiltfilt),
    preserving onset timing without introducing phase distortion.
    Explicitly documented as an offline research preprocessing operation.
    """
    
    def __init__(self, lowcut=0.5, highcut=40.0, notch_freq=60.0, notch_q=30.0, sfreq=256.0, order=4):
        self.sfreq = sfreq
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.notch_q = notch_q
        self.order = order
        
        # Bandpass SOS design
        self.sos_bp = signal.butter(order, [lowcut, highcut], btype="bandpass", fs=sfreq, output="sos")
        # Notch IIR design
        self.b_notch, self.a_notch = signal.iirnotch(notch_freq, notch_q, fs=sfreq)
        
    def filter_signal(self, data):
        """
        Filters EEG data array (channels, time) along the time axis.
        """
        # Apply bandpass (zero-phase SOS filter)
        filtered = signal.sosfiltfilt(self.sos_bp, data, axis=-1)
        # Apply 60 Hz notch filter (zero-phase IIR)
        filtered = signal.filtfilt(self.b_notch, self.a_notch, filtered, axis=-1)
        return filtered.astype(np.float32)


class CHBMITNormalizer:
    """
    Leakage-safe normalization.
    Ensures test and validation patient statistics are NEVER used during normalization fitting.
    Supports per-channel z-score (recording-local or fold-specific).
    """
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        
    def zscore_recording_local(self, data):
        """
        Per-channel z-score normalization local to a single recording.
        data: (channels, time)
        """
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        std = np.where(std < self.epsilon, 1.0, std)
        return (data - mean) / std
        
    def fit_fold_stats(self, training_data_list):
        """
        Calculates channel-wise mean and std across training fold only.
        """
        # Placeholder for fold-specific statistics fitting
        pass


class CHBMITStreamReader:
    """
    Memory-safe lazy streaming reader for CHB-MIT EDF recordings.
    Reads arbitrary time slices for the canonical 23 channels without loading full files into RAM.
    """
    
    def __init__(self, channel_manager=None, signal_filter=None):
        self.channel_manager = channel_manager or CHBMITChannelManager()
        self.signal_filter = signal_filter or CHBMITSignalFilter()
        
    def read_window(self, edf_path, start_sample, duration_samples=1280, apply_filter=True, apply_norm=True):
        """
        Reads a window slice from the specified EDF file.
        Returns: (data_tensor, actual_channels_count, montage_status)
        """
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        picked_indices, montage_status, missing = self.channel_manager.map_recording_channels(raw.ch_names)
        
        if len(picked_indices) == 0:
            raise ValueError(f"No valid channels found in {edf_path}")
            
        end_sample = start_sample + duration_samples
        data, _ = raw[picked_indices, start_sample:end_sample]
        
        if apply_filter:
            data = self.signal_filter.filter_signal(data)
            
        if apply_norm:
            norm = CHBMITNormalizer()
            data = norm.zscore_recording_local(data)
            
        return data, len(picked_indices), montage_status
