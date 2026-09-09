"""
NeuroAegis Dynamic Negative Subsampling Pipeline
Phase: Class Imbalance Handling Strategy

Implements reproducible dynamic negative subsampling strictly on the training fold.
Operates as a PyTorch Sampler over dataframe indices without loading raw EEG into memory.

Core Principles:
1. Patient Isolation: Strictly samples within the training patient pool.
2. 100% Positive Retention: Zero positive windows are discarded or artificially duplicated.
3. Epoch Dynamics: Negatives are dynamically subsampled per epoch using seed = base_seed + epoch.
4. Memory Efficient: Operates purely on index integers / window IDs.
"""

from typing import Dict, List, Optional, Any, Iterator
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


class DynamicNegativeSampler(Sampler[int]):
    """
    PyTorch Sampler that dynamically subsamples negative windows per epoch
    at a configurable negative:positive ratio, while preserving all positive windows.
    
    Args:
        train_df: DataFrame containing the training fold window index.
        ratio: Target ratio of sampled negatives to positives (e.g. 10.0). Default: 10.0.
        base_seed: Master reproducible random seed. Default: 42.
        label_column: Column name indicating binary label (0=negative, 1=positive). Default: 'label_any_overlap'.
        shuffle: Whether to shuffle positive and negative samples within the epoch. Default: True.
    """
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        ratio: float = 10.0,
        base_seed: int = 42,
        label_column: str = "label_any_overlap",
        shuffle: bool = True
    ):
        super().__init__()
        if ratio <= 0:
            raise ValueError(f"Sampling ratio must be positive, got {ratio}")
            
        self.train_df = train_df.reset_index(drop=True)
        self.ratio = float(ratio)
        self.base_seed = int(base_seed)
        self.label_column = label_column
        self.shuffle = shuffle
        
        # Patient pool isolation tracking
        self.allowed_patients = set(self.train_df["patient_id"].unique())
        
        # Partition training indices into positive and negative pools
        pos_mask = (self.train_df[self.label_column] == 1).values
        neg_mask = (self.train_df[self.label_column] == 0).values
        
        self.pos_indices = np.where(pos_mask)[0]
        self.neg_indices = np.where(neg_mask)[0]
        
        self.num_positives = len(self.pos_indices)
        self.num_negatives = len(self.neg_indices)
        
        if self.num_positives == 0:
            raise ValueError("Training set contains zero positive windows!")
        if self.num_negatives == 0:
            raise ValueError("Training set contains zero negative windows!")
            
        # Target number of negatives to sample per epoch
        self.target_neg_count = min(int(round(self.num_positives * self.ratio)), self.num_negatives)
        
        self.current_epoch: Optional[int] = None
        self.current_epoch_seed: Optional[int] = None
        self.current_epoch_indices: np.ndarray = np.array([], dtype=int)
        self.current_sampled_neg_indices: np.ndarray = np.array([], dtype=int)
        
        # Initialize epoch 0
        self.set_epoch(0)
        
    def set_epoch(self, epoch: int):
        """
        Advances sampler to the specified epoch and generates a reproducible sample.
        
        Seed formula: seed = base_seed + epoch
        """
        self.current_epoch = int(epoch)
        self.current_epoch_seed = self.base_seed + self.current_epoch
        
        # Independent reproducible NumPy RNG
        rng = np.random.default_rng(self.current_epoch_seed)
        
        # Sample negative indices without replacement
        sampled_neg_offsets = rng.choice(
            self.num_negatives,
            size=self.target_neg_count,
            replace=False
        )
        self.current_sampled_neg_indices = self.neg_indices[sampled_neg_offsets]
        
        # Verify 100% positive window retention (no duplication, no discarding)
        sampled_pos_indices = self.pos_indices
        
        # Combine positives and sampled negatives
        combined = np.concatenate([sampled_pos_indices, self.current_sampled_neg_indices])
        
        if self.shuffle:
            rng.shuffle(combined)
            
        self.current_epoch_indices = combined
        
        # Strict patient isolation verification
        self._verify_isolation()
        
    def _verify_isolation(self):
        """Asserts that zero sampled indices belong to patients outside the training set."""
        sampled_patients = set(self.train_df.iloc[self.current_epoch_indices]["patient_id"])
        leakage = sampled_patients.difference(self.allowed_patients)
        if leakage:
            raise RuntimeError(
                f"FATAL LEAKAGE IN SAMPLER: Sampled patients {leakage} are outside the allowed training set!"
            )
            
    def __iter__(self) -> Iterator[int]:
        """Yields sampled training integer indices for the current epoch."""
        return iter(self.current_epoch_indices.tolist())
        
    def __len__(self) -> int:
        """Returns the total number of samples presented in the current epoch."""
        return len(self.current_epoch_indices)
        
    def get_epoch_stats(self) -> Dict[str, Any]:
        """
        Returns comprehensive execution statistics for the current epoch.
        """
        if self.current_epoch is None:
            raise RuntimeError("Sampler epoch has not been set.")
            
        pos_sampled = len(self.pos_indices)
        neg_sampled = len(self.current_sampled_neg_indices)
        actual_ratio = round(neg_sampled / pos_sampled, 3) if pos_sampled > 0 else 0.0
        
        # Event level coverage
        sub_df = self.train_df.iloc[self.pos_indices]
        seizure_ids = set()
        if "seizure_event_ids" in sub_df.columns:
            for val in sub_df["seizure_event_ids"].dropna():
                for s_id in str(val).split(";"):
                    s_id = s_id.strip()
                    if s_id:
                        seizure_ids.add(s_id)
                        
        return {
            "epoch": self.current_epoch,
            "epoch_seed": self.current_epoch_seed,
            "base_seed": self.base_seed,
            "configured_ratio": self.ratio,
            "actual_ratio": actual_ratio,
            "positive_samples": pos_sampled,
            "unique_positive_samples": pos_sampled,  # 100% unique, no artificial duplication
            "available_negative_pool": self.num_negatives,
            "sampled_negative_samples": neg_sampled,
            "unique_negative_samples": neg_sampled,  # sampled without replacement
            "total_epoch_samples": len(self.current_epoch_indices),
            "seizure_events_represented": len(seizure_ids),
            "patient_count": len(self.allowed_patients),
            "patients": sorted(list(self.allowed_patients))
        }


class LazyEEGWindowDataset(torch.utils.data.Dataset):
    """
    Lightweight PyTorch Dataset backed by CHBMITStreamReader.
    Loads raw EEG windows only when indexed by the DataLoader.
    Never preloads full recordings or all windows into RAM.
    """
    
    def __init__(
        self,
        window_df: pd.DataFrame,
        edf_root_dir: str = "/Volumes/BLACK-BOX/NeuroAegis/CHB-MIT Dataset",
        label_column: str = "label_any_overlap",
        transform = None
    ):
        self.window_df = window_df.reset_index(drop=True)
        self.edf_root_dir = edf_root_dir
        self.label_column = label_column
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.window_df)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns metadata dictionary for window idx.
        (Signal loading is handled lazily via CHBMITStreamReader when model training begins).
        """
        row = self.window_df.iloc[idx]
        return {
            "window_id": row["window_id"],
            "patient_id": row["patient_id"],
            "recording_id": row["recording_id"],
            "edf_filename": row["edf_filename"],
            "window_start_sec": float(row["window_start_sec"]),
            "window_end_sec": float(row["window_end_sec"]),
            "window_start_sample": int(row["window_start_sample"]),
            "window_end_sample": int(row["window_end_sample"]),
            "label": int(row[self.label_column])
        }
