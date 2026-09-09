"""
NeuroAegis Phase 3: High-Performance Sequential CHB-MIT Data Pipeline
Memory-safe, deadlock-free window extraction and streaming evaluation engine
for training the 1D CNN baseline with Dynamic Negative Subsampling (10:1) and Binary Focal Loss.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import mne
mne.set_log_level("ERROR")

from research.phase_2.chbmit_preprocessor import (
    CHBMITChannelManager,
    CHBMITSignalFilter,
    CHBMITNormalizer
)
from research.imbalance.focal_loss import logits_to_probabilities


class CHBMITDataPipeline:
    """
    Thread-safe, memory-safe data pipeline for CHB-MIT EEG windows.
    Reads canonical 23 picked channels sequentially to eliminate thread contention.
    """
    def __init__(
        self,
        edf_root_dir: str = "/Volumes/BLACK-BOX/NeuroAegis/CHB-MIT Dataset",
        window_samples: int = 1280
    ):
        self.edf_root_dir = edf_root_dir
        self.window_samples = window_samples
        self.channel_manager = CHBMITChannelManager()
        self.signal_filter = CHBMITSignalFilter()
        self.normalizer = CHBMITNormalizer()
        
    def extract_single_recording_windows(
        self,
        patient_id: str,
        edf_filename: str,
        start_samples: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        edf_path = os.path.join(self.edf_root_dir, patient_id, edf_filename)
        if not os.path.exists(edf_path):
            raise FileNotFoundError(f"EDF file not found: {edf_path}")
            
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        picks, status, missing = self.channel_manager.map_recording_channels(raw.ch_names)
        
        n_wins = len(start_samples)
        windows = np.zeros((n_wins, 23, self.window_samples), dtype=np.float32)
        
        for i, s in enumerate(start_samples):
            s = int(s)
            e = s + self.window_samples
            try:
                data, _ = raw[picks, s:e]
                if data.shape[0] < 23:
                    padded = np.zeros((23, self.window_samples), dtype=np.float32)
                    padded[:data.shape[0], :data.shape[1]] = data
                    data = padded
                data = self.signal_filter.filter_signal(data)
                data = self.normalizer.zscore_recording_local(data)
                windows[i] = data.astype(np.float32)
            except Exception:
                pass
                
        return windows, labels.astype(np.float32)

    def load_epoch_windows(
        self,
        df: pd.DataFrame,
        sampled_indices: np.ndarray,
        label_column: str = "label_50pct_overlap",
        shuffle: bool = True,
        seed: Optional[int] = None,
        progress_callback = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sub_df = df.iloc[sampled_indices].copy()
        grouped = list(sub_df.groupby(["patient_id", "edf_filename"]))
        total_groups = len(grouped)
        
        all_x = []
        all_y = []
        
        for idx, ((pat_id, edf_file), group) in enumerate(grouped):
            starts = group["window_start_sample"].values
            lbls = group[label_column].values
            x_rec, y_rec = self.extract_single_recording_windows(pat_id, edf_file, starts, lbls)
            if len(x_rec) > 0:
                all_x.append(x_rec)
                all_y.append(y_rec)
            if progress_callback and (idx + 1) % 50 == 0:
                progress_callback(idx + 1, total_groups)
                
        X = np.concatenate(all_x, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        if shuffle:
            rng = np.random.default_rng(seed if seed is not None else 42)
            perm = rng.permutation(len(X))
            X = X[perm]
            y = y[perm]
            
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()

    def evaluate_split(
        self,
        model: torch.nn.Module,
        split_df: pd.DataFrame,
        device: torch.device,
        label_column: str = "label_50pct_overlap",
        batch_size: int = 256,
        progress_callback = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_probs = []
        all_trues = []
        
        grouped = list(split_df.groupby(["patient_id", "recording_id", "edf_filename"], sort=False))
        total_recs = len(grouped)
        
        with torch.no_grad():
            for rec_idx, ((pat_id, rec_id, edf_file), group) in enumerate(grouped):
                edf_path = os.path.join(self.edf_root_dir, pat_id, edf_file)
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                picks, status, missing = self.channel_manager.map_recording_channels(raw.ch_names)
                
                rec_data = raw.get_data(picks=picks)
                if rec_data.shape[0] < 23:
                    padded = np.zeros((23, rec_data.shape[1]), dtype=np.float32)
                    padded[:rec_data.shape[0], :] = rec_data
                    rec_data = padded
                    
                rec_data = self.signal_filter.filter_signal(rec_data)
                rec_data = self.normalizer.zscore_recording_local(rec_data)
                
                starts = group["window_start_sample"].values
                lbls = group[label_column].values
                n_win = len(starts)
                
                win_tensor = np.zeros((n_win, 23, self.window_samples), dtype=np.float32)
                for w_i, s in enumerate(starts):
                    win_tensor[w_i] = rec_data[:, s:s+self.window_samples]
                    
                t_input = torch.from_numpy(win_tensor).float()
                
                rec_probs = []
                for b_start in range(0, n_win, batch_size):
                    b_tensor = t_input[b_start:b_start+batch_size].to(device)
                    logits = model(b_tensor)
                    probs = logits_to_probabilities(logits).cpu().numpy()
                    rec_probs.append(probs)
                    
                if rec_probs:
                    all_probs.append(np.concatenate(rec_probs))
                    all_trues.append(lbls)
                    
                if progress_callback and (rec_idx + 1) % 10 == 0:
                    progress_callback(rec_idx + 1, total_recs, rec_id)
                    
        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_trues)
        return y_true, y_prob
