"""
NeuroAegis Phase 5: Raw EEG Sequence Loader
Efficient on-demand extraction of preprocessed EEG windows and causal 8-window sequences
from raw EDF files using the exact frozen preprocessing pipeline.
"""

import os
import sys
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import mne
mne.set_log_level("ERROR")

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from research.experiments.windowing_labeling.chbmit_preprocessor import (
    CHBMITChannelManager,
    CHBMITSignalFilter,
    CHBMITNormalizer
)


class RawEEGLoader:
    """
    Loads raw EDF files, applies frozen filtering and z-scoring,
    and extracts causal sequences of windows without redundant disk IO.
    """
    def __init__(
        self,
        edf_root_dir: str = os.path.join(BASE_DIR, "CHB-MIT Dataset")
    ):
        self.edf_root_dir = edf_root_dir
        self.channel_manager = CHBMITChannelManager()
        self.signal_filter = CHBMITSignalFilter()
        self.normalizer = CHBMITNormalizer()
        
        # In-memory cache for currently active recording
        self._cached_recording_id: Optional[str] = None
        self._cached_windows: Optional[torch.Tensor] = None

    def load_recording_windows(self, patient_id: str, recording_id: str) -> torch.Tensor:
        """
        Loads and preprocesses an entire recording into a tensor of windows:
        Shape: (N_windows, 23, 1280)
        """
        if self._cached_recording_id == recording_id and self._cached_windows is not None:
            return self._cached_windows

        edf_filename = f"{recording_id}.edf"
        edf_path = os.path.join(self.edf_root_dir, patient_id, edf_filename)
        if not os.path.exists(edf_path):
            raise FileNotFoundError(f"EDF file not found: {edf_path}")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        picks, status, missing = self.channel_manager.map_recording_channels(raw.ch_names)
        
        clean_names = [ch.strip().upper().replace(".", "") for ch in raw.ch_names]
        name_to_idx = {name: idx for idx, name in enumerate(clean_names)}

        if len(picks) > 0 and status != "COMMON_REF_CS2":
            rec_data = raw.get_data(picks=picks)
            if rec_data.shape[0] < 23:
                padded = np.zeros((23, rec_data.shape[1]), dtype=np.float32)
                padded[:rec_data.shape[0], :] = rec_data
                rec_data = padded
        elif status == "COMMON_REF_CS2" or any("-CS2" in name for name in clean_names):
            raw_data = raw.get_data()
            rec_data = np.zeros((23, raw_data.shape[1]), dtype=np.float32)
            for ch_idx, target_ch in enumerate(self.channel_manager.canonical_channels):
                parts = target_ch.split("-")
                if len(parts) == 2:
                    c1 = f"{parts[0]}-CS2"
                    c2 = f"{parts[1]}-CS2"
                    if c1 in name_to_idx and c2 in name_to_idx:
                        rec_data[ch_idx] = raw_data[name_to_idx[c1]] - raw_data[name_to_idx[c2]]
        else:
            raw_data = raw.get_data()
            rec_data = np.zeros((23, raw_data.shape[1]), dtype=np.float32)
            for ch_idx, pick in enumerate(picks[:23]):
                rec_data[ch_idx] = raw_data[pick]

        # 0.5-40 Hz Butterworth + 60 Hz notch
        rec_data = self.signal_filter.filter_signal(rec_data)
        # Per-channel local z-score
        rec_data = self.normalizer.zscore_recording_local(rec_data)

        t_data = torch.from_numpy(rec_data.astype(np.float32))
        windows = t_data.unfold(dimension=1, size=1280, step=640).permute(1, 0, 2).contiguous()
        
        self._cached_recording_id = recording_id
        self._cached_windows = windows
        return windows

    def extract_causal_sequence(
        self,
        patient_id: str,
        recording_id: str,
        target_window_index: int,
        seq_len: int = 8
    ) -> torch.Tensor:
        """
        Extracts an 8-window causal sequence ending at target_window_index.
        Applies causal zero left-padding if target_window_index < seq_len - 1.
        
        Returns:
            Tensor of shape (1, seq_len, 23, 1280)
        """
        windows = self.load_recording_windows(patient_id, recording_id)
        n_wins = windows.shape[0]
        
        assert 0 <= target_window_index < n_wins, f"Target index {target_window_index} out of bounds [0, {n_wins})"
        
        seq_tensor = torch.zeros((seq_len, 23, 1280), dtype=torch.float32)
        
        start_idx = target_window_index - seq_len + 1
        if start_idx >= 0:
            seq_tensor = windows[start_idx : target_window_index + 1].clone()
        else:
            # Left padding with zeros
            valid_wins = windows[0 : target_window_index + 1]
            pad_count = seq_len - len(valid_wins)
            seq_tensor[pad_count:] = valid_wins
            
        return seq_tensor.unsqueeze(0)  # (1, 8, 23, 1280)
