"""
NeuroAegis Phase 4B: Temporal Sequence Dataset & Builder
Constructs temporally-continuous EEG window sequences for causal GRU sequence modeling.

Core Principles:
1. Patient & Recording Isolation: Sequences are strictly built within each recording.
   No sequence crosses recording boundaries, patient boundaries, or train/val/test splits.
2. Temporal Continuity: Windows within a sequence must have exactly expected stride (2.5s).
   Any recording gap terminates the sequence history.
3. Causal History & Left-Padding: Sequences use only current and past windows.
   Start-of-recording windows with < L history are causally left-padded with zero embeddings.
4. Target Labeling: The sequence label strictly matches the target (final/current) window
   using the frozen primary label: label_50pct_overlap.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Iterator
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class SequenceBuilder:
    """
    Constructs temporally contiguous sequence metadata from a split's window index DataFrame.
    """
    def __init__(
        self,
        window_duration_sec: float = 5.0,
        window_stride_sec: float = 2.5,
        label_column: str = "label_50pct_overlap",
        padding_mode: str = "causal_zero_left_padding"
    ):
        self.window_duration_sec = float(window_duration_sec)
        self.window_stride_sec = float(window_stride_sec)
        self.label_column = label_column
        self.padding_mode = padding_mode

    def compute_temporal_span(self, seq_len: int) -> float:
        """Computes exact temporal span in seconds: duration + (L - 1) * stride."""
        if seq_len <= 0:
            raise ValueError(f"Sequence length must be positive, got {seq_len}")
        return self.window_duration_sec + (seq_len - 1) * self.window_stride_sec

    def build_sequences(self, split_df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
        """
        Builds all valid sequence definitions for a split at sequence length L.
        
        Args:
            split_df: Window index dataframe for a single split (train, val, or test).
            seq_len: Sequence length L in {1, 4, 8, 12}.
            
        Returns:
            DataFrame where each row represents one sequence targeting a specific window.
        """
        if seq_len <= 0:
            raise ValueError(f"Sequence length must be positive, got {seq_len}")

        df = split_df.reset_index(drop=True)
        temporal_span = self.compute_temporal_span(seq_len)

        grouped = df.groupby(["patient_id", "recording_id"], sort=False)

        seq_records = []
        
        for (pat_id, rec_id), group in grouped:
            sorted_group = group.sort_values("window_start_sec")
            group_indices = sorted_group.index.values
            start_secs = sorted_group["window_start_sec"].values
            labels = sorted_group[self.label_column].values
            window_ids = sorted_group["window_id"].values
            
            n_win = len(sorted_group)
            if n_win == 0:
                continue

            for i in range(n_win):
                curr_split_idx = group_indices[i]
                curr_start = start_secs[i]
                curr_label = int(labels[i])
                curr_win_id = window_ids[i]

                seq_window_indices = [curr_split_idx]
                
                if seq_len > 1:
                    for step in range(1, seq_len):
                        prev_pos = i - step
                        if prev_pos < 0:
                            seq_window_indices.insert(0, -1)
                        else:
                            expected_start = curr_start - step * self.window_stride_sec
                            actual_start = start_secs[prev_pos]
                            
                            if abs(actual_start - expected_start) < 0.05:
                                seq_window_indices.insert(0, group_indices[prev_pos])
                            else:
                                seq_window_indices.insert(0, -1)
                
                seq_records.append({
                    "seq_id": f"{curr_win_id}_L{seq_len:02d}",
                    "patient_id": pat_id,
                    "recording_id": rec_id,
                    "target_window_id": curr_win_id,
                    "target_split_idx": curr_split_idx,
                    "target_start_sec": curr_start,
                    "target_label": curr_label,
                    "seq_len": seq_len,
                    "temporal_span_sec": temporal_span,
                    "window_indices": seq_window_indices
                })

        seq_df = pd.DataFrame(seq_records)
        return seq_df


class SequenceDataset(Dataset):
    """
    PyTorch Dataset providing sequence embeddings for training and evaluation.
    Operates on precomputed (N_windows, embedding_dim) embeddings.
    """
    def __init__(
        self,
        sequence_df: pd.DataFrame,
        embeddings: np.ndarray,
        embedding_dim: int = 128
    ):
        self.sequence_df = sequence_df.reset_index(drop=True)
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.zero_padding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        self.labels = self.sequence_df["target_label"].values.astype(np.float32)
        self.window_indices_matrix = np.array(self.sequence_df["window_indices"].tolist(), dtype=np.int32)
        self.seq_len = self.window_indices_matrix.shape[1]

    def __len__(self) -> int:
        return len(self.sequence_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.window_indices_matrix[idx]
        seq_tensor = np.zeros((self.seq_len, self.embedding_dim), dtype=np.float32)
        
        for step, w_idx in enumerate(indices):
            if w_idx >= 0:
                seq_tensor[step] = self.embeddings[w_idx]
            else:
                seq_tensor[step] = self.zero_padding

        return torch.from_numpy(seq_tensor), torch.tensor(self.labels[idx], dtype=torch.float32)


class SequenceNegativeSampler(Sampler[int]):
    """
    Dynamic negative subsampler for sequence data.
    Ensures 100% positive sequence retention, 10:1 negative sampling per epoch,
    reproducible random seed (base_seed + epoch), and zero patient leakage.
    """
    def __init__(
        self,
        sequence_df: pd.DataFrame,
        ratio: float = 10.0,
        base_seed: int = 42,
        shuffle: bool = True
    ):
        super().__init__()
        self.sequence_df = sequence_df.reset_index(drop=True)
        self.ratio = float(ratio)
        self.base_seed = int(base_seed)
        self.shuffle = shuffle
        
        self.allowed_patients = set(self.sequence_df["patient_id"].unique())
        
        labels = self.sequence_df["target_label"].values
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]
        
        self.num_positives = len(self.pos_indices)
        self.num_negatives = len(self.neg_indices)
        
        if self.num_positives == 0:
            raise ValueError("Sequence set contains zero positive sequences!")
        if self.num_negatives == 0:
            raise ValueError("Sequence set contains zero negative sequences!")
            
        self.target_neg_count = min(int(round(self.num_positives * self.ratio)), self.num_negatives)
        
        self.current_epoch: Optional[int] = None
        self.current_epoch_seed: Optional[int] = None
        self.current_epoch_indices: np.ndarray = np.array([], dtype=int)
        self.current_sampled_neg_indices: np.ndarray = np.array([], dtype=int)
        
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        self.current_epoch_seed = self.base_seed + self.current_epoch
        
        rng = np.random.default_rng(self.current_epoch_seed)
        
        sampled_neg_offsets = rng.choice(
            self.num_negatives,
            size=self.target_neg_count,
            replace=False
        )
        self.current_sampled_neg_indices = self.neg_indices[sampled_neg_offsets]
        
        combined = np.concatenate([self.pos_indices, self.current_sampled_neg_indices])
        
        if self.shuffle:
            rng.shuffle(combined)
            
        self.current_epoch_indices = combined
        self._verify_isolation()

    def _verify_isolation(self):
        sampled_patients = set(self.sequence_df.iloc[self.current_epoch_indices]["patient_id"])
        leakage = sampled_patients.difference(self.allowed_patients)
        if leakage:
            raise RuntimeError(f"FATAL LEAKAGE IN SEQUENCE SAMPLER: Sampled patients {leakage} outside allowed pool!")

    def __iter__(self) -> Iterator[int]:
        return iter(self.current_epoch_indices.tolist())

    def __len__(self) -> int:
        return len(self.current_epoch_indices)

    def get_epoch_stats(self) -> Dict[str, Any]:
        return {
            "epoch": self.current_epoch,
            "epoch_seed": self.current_epoch_seed,
            "base_seed": self.base_seed,
            "sampling_ratio": self.ratio,
            "positive_sequences": len(self.pos_indices),
            "sampled_negative_sequences": len(self.current_sampled_neg_indices),
            "total_epoch_sequences": len(self.current_epoch_indices),
            "patient_count": len(self.allowed_patients)
        }
