"""
NeuroAegis Phase 4B: Incremental Embedding Cache Manager
Caches 128-dimensional spatial embeddings from the frozen CNN + GNN backbone on disk.
Ensures zero redundant computation, fast memory-mapped loading, and crash resilience.
"""

import os
import sys
import time
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import torch
import mne
mne.set_log_level("ERROR")

from research.experiments.windowing_labeling.chbmit_preprocessor import (
    CHBMITChannelManager,
    CHBMITSignalFilter,
    CHBMITNormalizer
)
from neuroaegis.models.baselines.model_c import CNN_GNN_GRU


class EmbeddingCacheManager:
    """
    Manages computation and disk caching of 128-d spatial embeddings per recording.
    """
    def __init__(
        self,
        cache_dir: str = "research/experiments/model_c/embeddings_cache",
        edf_root_dir: str = "CHB-MIT Dataset",
        frozen_backbone_path: str = "artifacts/checkpoints/frozen_cnn_gnn.pt",
        batch_size: int = 256,
        device: Optional[torch.device] = None
    ):
        self.cache_dir = cache_dir
        self.edf_root_dir = edf_root_dir
        self.batch_size = batch_size
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.channel_manager = CHBMITChannelManager()
        self.signal_filter = CHBMITSignalFilter()
        self.normalizer = CHBMITNormalizer()
        
        self.model = CNN_GNN_GRU(frozen_backbone_path=frozen_backbone_path, device=self.device)
        self.model.eval()

    def get_cache_path(self, patient_id: str, recording_id: str) -> str:
        pat_dir = os.path.join(self.cache_dir, patient_id)
        os.makedirs(pat_dir, exist_ok=True)
        return os.path.join(pat_dir, f"{recording_id}.npy")

    def has_cached(self, patient_id: str, recording_id: str) -> bool:
        path = self.get_cache_path(patient_id, recording_id)
        return os.path.exists(path) and os.path.getsize(path) > 1024

    def compute_recording_embeddings(
        self,
        patient_id: str,
        recording_id: str,
        edf_filename: str,
        expected_windows: Optional[int] = None
    ) -> np.ndarray:
        """
        Loads raw EDF, preprocesses signal, extracts windows via zero-copy unfold,
        runs through frozen CNN+GNN, and saves to disk cache.
        """
        cache_path = self.get_cache_path(patient_id, recording_id)
        if self.has_cached(patient_id, recording_id):
            return np.load(cache_path)

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
            # Derive bipolar channels from common CS2 reference
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

        rec_data = self.signal_filter.filter_signal(rec_data)
        rec_data = self.normalizer.zscore_recording_local(rec_data)

        t_data = torch.from_numpy(rec_data.astype(np.float32))
        windows = t_data.unfold(dimension=1, size=1280, step=640).permute(1, 0, 2).contiguous()
        n_wins = windows.shape[0]

        if expected_windows is not None and n_wins != expected_windows:
            # If slight rounding differences occur at recording end, adjust
            if n_wins > expected_windows:
                windows = windows[:expected_windows]
                n_wins = expected_windows

        embs = []
        with torch.no_grad():
            for b in range(0, n_wins, self.batch_size):
                batch = windows[b:b+self.batch_size].to(self.device)
                emb = self.model.extract_backbone_embedding(batch).cpu()
                embs.append(emb)
                del batch

        if len(embs) > 0:
            rec_embs = torch.cat(embs, dim=0).numpy().astype(np.float32)
        else:
            rec_embs = np.zeros((0, 128), dtype=np.float32)

        del raw, rec_data, t_data, windows, embs
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Atomic write to disk
        temp_path = cache_path.replace(".npy", "_tmp.npy")
        np.save(temp_path, rec_embs)
        os.replace(temp_path, cache_path)

        return rec_embs

    def get_split_embeddings(
        self,
        split_df: pd.DataFrame,
        split_name: str = "split",
        progress_callback = None
    ) -> np.ndarray:
        """
        Loads or computes embeddings for an entire split in exact DataFrame order.
        Returns a single unified array of shape (N_windows, 128).
        """
        unified_path = os.path.join(self.cache_dir, f"{split_name}_embeddings_unified.npy")
        if os.path.exists(unified_path) and os.path.getsize(unified_path) > 1024:
            arr = np.load(unified_path, mmap_mode="r")
            if len(arr) == len(split_df):
                return arr

        grouped = list(split_df.groupby(["patient_id", "recording_id", "edf_filename"], sort=False))
        total_recs = len(grouped)

        all_embs = []
        for rec_idx, ((pat_id, rec_id, edf_file), group) in enumerate(grouped):
            expected_n = len(group)
            rec_emb = self.compute_recording_embeddings(pat_id, rec_id, edf_file, expected_windows=expected_n)
            
            if len(rec_emb) != expected_n:
                if len(rec_emb) > expected_n:
                    rec_emb = rec_emb[:expected_n]
                else:
                    pad = np.zeros((expected_n - len(rec_emb), 128), dtype=np.float32)
                    rec_emb = np.concatenate([rec_emb, pad], axis=0)

            all_embs.append(rec_emb)
            if progress_callback:
                progress_callback(rec_idx + 1, total_recs, rec_id)

        unified_embs = np.concatenate(all_embs, axis=0).astype(np.float32)
        del all_embs
        import gc
        gc.collect()
        
        # Save unified array
        temp_path = unified_path.replace(".npy", "_tmp.npy")
        np.save(temp_path, unified_embs)
        os.replace(temp_path, unified_path)

        return unified_embs

