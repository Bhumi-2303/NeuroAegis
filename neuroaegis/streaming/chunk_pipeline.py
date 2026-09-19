"""
neuroaegis/streaming/chunk_pipeline.py
───────────────────────────────────────
Bounded Chunked Streaming Engine for CHB-MIT EEG.

Eliminates full-dataset window materialization:
  - The dataset is NEVER represented as (N, 23, 1280) in RAM.
  - Slices EEG segments in bounded chunks (initial: 512, max: 1024 windows).
  - Preprocesses (notch + bandpass + local z-score) on the bounded chunk.
  - Trains or evaluates in mini-batches (batch_size=4).
  - Streams prediction outputs directly to disk.
  - Releases chunk memory and flushes MPS/CPU cache before reading the next chunk.
  - Dynamically reduces chunk size (1024 -> 512 -> 256 -> 128) if RSS approaches 4 GB.
"""

import os
import sys
import gc
import csv
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Generator, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mne

mne.set_log_level("ERROR")

from neuroaegis.utils.memory import (
    flush_memory,
    get_live_rss_mb,
    get_peak_rss_mb,
)
from research.phase_2.chbmit_preprocessor import (
    CHBMITChannelManager,
    CHBMITSignalFilter,
    CHBMITNormalizer,
)
from research.imbalance.focal_loss import logits_to_probabilities


@dataclass
class BoundedChunkConfig:
    """Configuration for bounded chunked streaming pipeline."""
    initial_chunk_size: int = 512
    max_chunk_size: int = 1024
    min_chunk_size: int = 128
    batch_size: int = 4
    device: str = "mps"
    window_samples: int = 1280  # 5.0 seconds @ 256 Hz
    window_stride_samples: int = 640  # 2.5 seconds @ 256 Hz (50% overlap)
    rss_threshold_mb: float = 4000.0  # 4 GB safety ceiling
    rss_preferred_mb: float = 3000.0  # 3 GB preferred ceiling
    current_chunk_size: int = 512

    def adapt_chunk_size_if_needed(self) -> int:
        """Adapts chunk size if memory pressure is detected."""
        peak_rss = get_peak_rss_mb()
        if peak_rss > self.rss_threshold_mb:
            old_size = self.current_chunk_size
            if self.current_chunk_size > 512:
                self.current_chunk_size = 512
            elif self.current_chunk_size > 256:
                self.current_chunk_size = 256
            elif self.current_chunk_size > 128:
                self.current_chunk_size = 128
            print(f"[BoundedStreaming] Memory warning: Peak RSS={peak_rss:.1f} MB > {self.rss_threshold_mb} MB. "
                  f"Adapted chunk_size {old_size} -> {self.current_chunk_size}")
        return self.current_chunk_size


class IncrementalPredictionWriter:
    """
    Stream-aligned, incremental disk writer for prediction records.
    Appends predictions chunk-by-chunk directly to disk with zero RAM accumulation.
    Preserves exact alignment:
      patient_id, recording_id, edf_filename, window_start_sec,
      window_end_sec, window_start_sample, window_end_sample,
      label_50pct_overlap, prediction_prob
    """
    def __init__(self, output_filepath: Union[str, Path]):
        self.filepath = Path(output_filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.total_records = 0
        
        # Initialize file with header
        self.fieldnames = [
            "patient_id",
            "recording_id",
            "edf_filename",
            "window_start_sec",
            "window_end_sec",
            "window_start_sample",
            "window_end_sample",
            "label_50pct_overlap",
            "prediction_prob",
        ]
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.fieldnames)

    def write_chunk(self, meta_df: pd.DataFrame, probs: np.ndarray):
        """Appends a single chunk of predictions to disk."""
        assert len(meta_df) == len(probs), f"Length mismatch: meta {len(meta_df)} vs probs {len(probs)}"
        
        records = []
        patient_ids = meta_df["patient_id"].values
        recording_ids = meta_df["recording_id"].values if "recording_id" in meta_df.columns else patient_ids
        edf_files = meta_df["edf_filename"].values
        start_secs = meta_df["window_start_sec"].values if "window_start_sec" in meta_df.columns else [0.0]*len(meta_df)
        end_secs = meta_df["window_end_sec"].values if "window_end_sec" in meta_df.columns else [0.0]*len(meta_df)
        start_samples = meta_df["window_start_sample"].values
        end_samples = meta_df["window_end_sample"].values if "window_end_sample" in meta_df.columns else (start_samples + 1280)
        labels = meta_df["label_50pct_overlap"].values

        for i in range(len(probs)):
            records.append([
                patient_ids[i],
                recording_ids[i],
                edf_files[i],
                float(start_secs[i]),
                float(end_secs[i]),
                int(start_samples[i]),
                int(end_samples[i]),
                int(labels[i]),
                float(probs[i]),
            ])

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(records)

        self.total_records += len(records)
        del records


class BoundedStreamingDataPipeline:
    """
    Thread-safe, bounded chunked streaming pipeline for CHB-MIT EEG.
    Processes ≤ 1024 windows per chunk, yielding arrays that are immediately
    consumed and freed, guaranteeing bounded memory regardless of dataset size.
    """
    def __init__(
        self,
        edf_root_dir: str = "/Volumes/BLACK-BOX/NeuroAegis/CHB-MIT Dataset",
        config: Optional[BoundedChunkConfig] = None,
    ):
        self.edf_root_dir = Path(edf_root_dir)
        self.config = config or BoundedChunkConfig()
        self.channel_manager = CHBMITChannelManager()
        self.signal_filter = CHBMITSignalFilter()
        self.normalizer = CHBMITNormalizer()

    def stream_recording_chunks(
        self,
        patient_id: str,
        edf_filename: str,
        recording_window_df: pd.DataFrame,
        chunk_size: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, pd.DataFrame], None, None]:
        """
        Yields bounded chunks of preprocessed EEG windows from a single EDF recording.
        Each chunk is of shape (chunk_len, 23, window_samples) in float32.
        """
        active_chunk_size = chunk_size or self.config.current_chunk_size
        active_chunk_size = min(active_chunk_size, self.config.max_chunk_size)

        edf_path = self.edf_root_dir / patient_id / edf_filename
        if not edf_path.exists():
            raise FileNotFoundError(f"EDF file not found: {edf_path}")

        n_total_windows = len(recording_window_df)
        if n_total_windows == 0:
            return

        # Open lazy EDF file once for this recording
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        picks, status, missing = self.channel_manager.map_recording_channels(raw.ch_names)

        # Slices windows in bounded chunks
        for chunk_start in range(0, n_total_windows, active_chunk_size):
            chunk_df = recording_window_df.iloc[chunk_start : chunk_start + active_chunk_size]
            chunk_len = len(chunk_df)

            starts = chunk_df["window_start_sample"].values
            lbls = chunk_df["label_50pct_overlap"].values.astype(np.float32)

            min_s = int(np.min(starts))
            max_e = int(np.max(starts)) + self.config.window_samples

            # Read the required contiguous span for this chunk (at most ~30-60 MB)
            chunk_raw, _ = raw[picks, min_s:max_e]

            if chunk_raw.shape[0] < 23:
                padded = np.zeros((23, chunk_raw.shape[1]), dtype=np.float32)
                padded[:chunk_raw.shape[0], :] = chunk_raw
                chunk_raw = padded

            X_chunk = np.zeros((chunk_len, 23, self.config.window_samples), dtype=np.float32)

            for i, s in enumerate(starts):
                rel_s = int(s) - min_s
                rel_e = rel_s + self.config.window_samples
                data = chunk_raw[:, rel_s:rel_e]
                if data.shape[1] < self.config.window_samples:
                    pad = np.zeros((23, self.config.window_samples), dtype=np.float32)
                    pad[:, :data.shape[1]] = data
                    data = pad
                data = self.signal_filter.filter_signal(data)
                data = self.normalizer.zscore_recording_local(data)
                X_chunk[i] = data.astype(np.float32)

            del chunk_raw
            yield X_chunk, lbls, chunk_df

            # Release chunk array and invoke flush
            del X_chunk, lbls, chunk_df
            flush_memory()

        del raw
        flush_memory()

    def stream_dataset_chunks(
        self,
        dataset_df: pd.DataFrame,
        chunk_size: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, pd.DataFrame], None, None]:
        """
        Iterates over an entire dataset specification (regardless of length: 1,000 or 1,000,000 windows)
        grouping by recording and yielding bounded chunks without ever materializing the dataset in RAM.
        """
        active_chunk_size = chunk_size or self.config.current_chunk_size
        groups = dataset_df.groupby(["patient_id", "edf_filename"], sort=False)

        for (pat_id, edf_file), group_df in groups:
            yield from self.stream_recording_chunks(pat_id, edf_file, group_df, active_chunk_size)


class StreamingTrainer:
    """
    Executes model training over a stream of bounded window chunks.
    Performs mini-batch gradient updates (batch_size=4) and releases chunk buffers.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        batch_size: int = 4,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size

    def train_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Dict[str, float]:
        """
        Trains on a single bounded chunk (e.g. 512 windows) in mini-batches of batch_size=4.
        """
        self.model.train()
        n_samples = len(X_chunk)
        if n_samples == 0:
            return {"loss": 0.0, "steps": 0}

        t_x = torch.from_numpy(X_chunk)
        t_y = torch.from_numpy(y_chunk).unsqueeze(1)
        dataset = TensorDataset(t_x, t_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False)

        chunk_loss_sum = 0.0
        chunk_steps = 0

        for bx, by in loader:
            bx = bx.to(self.device)
            by = by.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(bx)
            loss = self.criterion(logits, by)
            loss.backward()
            self.optimizer.step()

            chunk_loss_sum += float(loss.item())
            chunk_steps += 1
            del bx, by, logits, loss

        del loader, dataset, t_x, t_y
        self.optimizer.zero_grad(set_to_none=True)
        flush_memory()

        avg_loss = chunk_loss_sum / max(chunk_steps, 1)
        return {"loss": avg_loss, "steps": chunk_steps, "samples": n_samples}


class StreamingEvaluator:
    """
    Executes gradient-free evaluation over a stream of bounded window chunks.
    Writes predictions incrementally to disk and computes scalar online metrics.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int = 4,
        prediction_writer: Optional[IncrementalPredictionWriter] = None,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.prediction_writer = prediction_writer

    def evaluate_chunk(
        self,
        X_chunk: np.ndarray,
        y_chunk: np.ndarray,
        meta_chunk: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Evaluates a single bounded chunk without accumulating tensors in memory.
        """
        self.model.eval()
        n_samples = len(X_chunk)
        if n_samples == 0:
            return {"samples": 0, "probs": []}

        t_x = torch.from_numpy(X_chunk)
        chunk_probs = []

        with torch.no_grad():
            for b_start in range(0, n_samples, self.batch_size):
                bx = t_x[b_start : b_start + self.batch_size].to(self.device)
                logits = self.model(bx)
                probs = logits_to_probabilities(logits).cpu().numpy().flatten()
                chunk_probs.append(probs)
                del bx, logits, probs

        del t_x
        flush_memory()

        all_probs = np.concatenate(chunk_probs) if chunk_probs else np.array([], dtype=np.float32)
        del chunk_probs

        if self.prediction_writer is not None:
            self.prediction_writer.write_chunk(meta_chunk, all_probs)

        # Compute scalar confusion stats
        preds_binary = (all_probs >= 0.5).astype(int)
        y_int = y_chunk.astype(int)

        tp = int(np.sum((preds_binary == 1) & (y_int == 1)))
        fp = int(np.sum((preds_binary == 1) & (y_int == 0)))
        tn = int(np.sum((preds_binary == 0) & (y_int == 0)))
        fn = int(np.sum((preds_binary == 0) & (y_int == 1)))

        del all_probs, preds_binary, y_int
        flush_memory()

        return {
            "samples": n_samples,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
