"""
siena_zero_shot_evaluator.py
─────────────────────────────
NeuroAegis Research — Phase 6: Cross-Dataset Generalization (Siena)
Evaluates the frozen Phase 4B CNN_GNN_GRU model directly on Siena Scalp EEG.
Strict Zero-Shot Protocol:
  - Model weights FROZEN (requires_grad = False)
  - Graph structure FROZEN (theta = 0.30, 23 nodes, 40 edges)
  - Decision threshold FROZEN (tau = 0.50)
  - Event alarm parameters FROZEN from Phase 4B:
      * Moving average smoothing window = 3
      * Minimum alarm duration = 3 consecutive windows (5.0s)
      * Merge interval = 10.0s
      * Detection tolerance = 10.0s pre/post
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from research.experiments.xai.xai.xai_model import XAIModelWrapper
from research.experiments.siena.siena_preprocessor import (
    SienaChannelHarmonizer,
    SienaPreprocessor,
    SienaWindowExtractor,
    SienaSequenceBuilder
)

RESULTS_DIR = os.path.join(BASE_DIR, "research/experiments/siena/results")
MANIFEST_DIR = os.path.join(BASE_DIR, "research/experiments/siena/manifests")
os.makedirs(RESULTS_DIR, exist_ok=True)


class SienaZeroShotEvaluator:
    def __init__(
        self,
        checkpoint_path: str = os.path.join(BASE_DIR, "artifacts/checkpoints/frozen_cnn_gnn_gru.pt"),
        events_path: str = os.path.join(MANIFEST_DIR, "siena_seizure_events.csv"),
        manifest_path: str = os.path.join(MANIFEST_DIR, "siena_manifest.csv"),
        tau_threshold: float = 0.50,
        device: Optional[torch.device] = None
    ):
        self.checkpoint_path = checkpoint_path
        self.events_path = events_path
        self.manifest_path = manifest_path
        self.tau_threshold = tau_threshold
        
        # Determine compute device (MPS for Apple M4 if available)
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"[Evaluator] Initializing with device: {self.device}")
        
        # Load frozen model wrapper
        self.model = XAIModelWrapper(device=self.device)
        self.model.eval()
        
        # Verify checkpoint hash
        self.checkpoint_hash = self._get_sha256(checkpoint_path)
        print(f"[Evaluator] Verified Phase 4B checkpoint SHA256: {self.checkpoint_hash}")
        
        # Load manifests
        self.df_events = pd.read_csv(events_path)
        self.df_manifest = pd.read_csv(manifest_path)
        
        # Initialize preprocessor components
        self.harmonizer = SienaChannelHarmonizer()
        self.preprocessor = SienaPreprocessor()
        self.extractor = SienaWindowExtractor()
        self.seq_builder = SienaSequenceBuilder(seq_len=8)
        
        # Frozen Event Detection Parameters (Phase 4B)
        self.smooth_window = 3
        self.min_alarm_consecutive = 3
        self.merge_interval_sec = 10.0
        self.tolerance_sec = 10.0

    @staticmethod
    def _get_sha256(filepath: str) -> str:
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    def evaluate_recording(
        self,
        recording_id: str,
        raw_eeg_data: np.ndarray,
        channel_names: List[str]
    ) -> Dict:
        """
        Processes raw referential EEG, extracts windows, runs inference, and computes metrics.
        
        Args:
            recording_id: e.g. 'PN00/PN00-4.edf'
            raw_eeg_data: (n_channels, n_samples) at 512 Hz
            channel_names: List of channel labels in raw EDF
            
        Returns:
            Dictionary of window predictions, alarms, and recording metrics.
        """
        t0 = time.time()
        
        # 1. Harmonize to 23 canonical bipolar channels
        bipolar_data = self.harmonizer.harmonize_channels(raw_eeg_data, channel_names)
        
        # 2. Decimate (512 -> 256 Hz), bandpass, notch, z-score normalize
        normalized_data = self.preprocessor.process_bipolar_data(bipolar_data)
        
        # 3. Extract 5s windows (stride 2.5s)
        windows, time_intervals = self.extractor.extract_windows(normalized_data)
        n_windows = len(windows)
        
        # 4. Construct causal L=8 sequences
        sequences = self.seq_builder.build_causal_sequences(windows)
        
        # 5. Model inference
        probs = []
        batch_size = 64
        with torch.no_grad():
            for b in range(0, n_windows, batch_size):
                b_seq = torch.from_numpy(sequences[b : b + batch_size]).to(self.device)
                logits = self.model.forward_differentiable(b_seq)
                p = torch.sigmoid(logits).cpu().numpy().flatten()
                probs.extend(p)
        probs = np.array(probs, dtype=np.float32)
        
        inference_time = time.time() - t0
        
        # 6. Label ground truth (Strategy B: >= 50% overlap = 2.5s)
        rec_events = self.df_events[self.df_events["recording_id"] == recording_id]
        labels = np.zeros(n_windows, dtype=np.int32)
        event_assignments = [-1] * n_windows
        
        for w_idx, (w_start, w_end) in enumerate(time_intervals):
            for _, ev in rec_events.iterrows():
                ev_start = ev["start_sec"]
                ev_end = ev["end_sec"]
                ov_s = max(w_start, ev_start)
                ov_e = min(w_end, ev_end)
                ov_dur = max(0.0, ov_e - ov_s)
                if ov_dur >= 2.5:
                    labels[w_idx] = 1
                    event_assignments[w_idx] = ev["seizure_id"]
                    break

        # 7. Apply Phase 4B Event Detection Protocol
        # A. Moving average smoothing (window = 3)
        if len(probs) >= self.smooth_window:
            smoothed_probs = np.convolve(probs, np.ones(self.smooth_window) / self.smooth_window, mode="same")
        else:
            smoothed_probs = probs.copy()
            
        # B. Threshold at tau = 0.50
        binary_pred = (smoothed_probs >= self.tau_threshold).astype(int)
        
        # C. Extract raw alarm clusters
        raw_alarms = []
        in_alarm = False
        alarm_start_w = 0
        
        for w_idx in range(n_windows):
            if binary_pred[w_idx] == 1 and not in_alarm:
                in_alarm = True
                alarm_start_w = w_idx
            elif binary_pred[w_idx] == 0 and in_alarm:
                in_alarm = False
                alarm_len = w_idx - alarm_start_w
                if alarm_len >= self.min_alarm_consecutive:
                    raw_alarms.append({
                        "start_sec": time_intervals[alarm_start_w][0],
                        "end_sec": time_intervals[w_idx - 1][1],
                        "start_w": alarm_start_w,
                        "end_w": w_idx - 1,
                        "max_prob": float(np.max(smoothed_probs[alarm_start_w:w_idx]))
                    })
        if in_alarm:
            alarm_len = n_windows - alarm_start_w
            if alarm_len >= self.min_alarm_consecutive:
                raw_alarms.append({
                    "start_sec": time_intervals[alarm_start_w][0],
                    "end_sec": time_intervals[-1][1],
                    "start_w": alarm_start_w,
                    "end_w": n_windows - 1,
                    "max_prob": float(np.max(smoothed_probs[alarm_start_w:]))
                })
                
        # D. Merge alarms closer than merge_interval_sec (10.0s)
        merged_alarms = []
        for al in raw_alarms:
            if not merged_alarms:
                merged_alarms.append(al)
            else:
                prev = merged_alarms[-1]
                if al["start_sec"] - prev["end_sec"] <= self.merge_interval_sec:
                    prev["end_sec"] = al["end_sec"]
                    prev["end_w"] = al["end_w"]
                    prev["max_prob"] = max(prev["max_prob"], al["max_prob"])
                else:
                    merged_alarms.append(al)

        # 8. Event Matching
        event_evals = []
        matched_alarm_indices = set()
        
        for _, ev in rec_events.iterrows():
            ev_id = ev["seizure_id"]
            ev_start = ev["start_sec"]
            ev_end = ev["end_sec"]
            ev_dur = ev["duration_sec"]
            
            # Tolerant search window
            search_start = ev_start - self.tolerance_sec
            search_end = ev_end + self.tolerance_sec
            
            detected = False
            first_alarm_time = None
            delay = None
            
            for a_idx, al in enumerate(merged_alarms):
                # Check overlap
                if max(al["start_sec"], search_start) <= min(al["end_sec"], search_end):
                    detected = True
                    matched_alarm_indices.add(a_idx)
                    if first_alarm_time is None or al["start_sec"] < first_alarm_time:
                        first_alarm_time = al["start_sec"]
                        
            if detected:
                delay = max(0.0, first_alarm_time - ev_start)
                
            event_evals.append({
                "patient_id": ev["patient_id"],
                "recording_id": recording_id,
                "seizure_id": ev_id,
                "cohort_split": ev["cohort_split"],
                "event_start_sec": ev_start,
                "event_end_sec": ev_end,
                "event_duration_sec": ev_dur,
                "detected": detected,
                "alarm_time_sec": first_alarm_time if detected else np.nan,
                "detection_delay_sec": delay if detected else np.nan
            })
            
        # 9. False Alarms: Merged alarms that did not match ANY seizure
        false_alarms = []
        for a_idx, al in enumerate(merged_alarms):
            if a_idx not in matched_alarm_indices:
                false_alarms.append(al)
                
        rec_dur_hours = (time_intervals[-1][1]) / 3600.0 if time_intervals else 0.0
        
        return {
            "recording_id": recording_id,
            "n_windows": n_windows,
            "duration_hours": rec_dur_hours,
            "inference_time_sec": inference_time,
            "probs": probs,
            "smoothed_probs": smoothed_probs,
            "labels": labels,
            "time_intervals": time_intervals,
            "event_assignments": event_assignments,
            "merged_alarms": merged_alarms,
            "event_evals": event_evals,
            "false_alarms": false_alarms,
            "n_false_alarms": len(false_alarms),
            "fa_per_24h": (len(false_alarms) / rec_dur_hours * 24.0) if rec_dur_hours > 0 else 0.0
        }
