"""
NeuroAegis Phase 4B: CNN + Spatial GNN + Temporal GRU Training & Ablation Engine
Trains and evaluates the causal GRU temporal sequence model on CHB-MIT EEG:
- Frozen Phase 4A-C CNN + GNN spatial backbone (θ=0.30 graph, 52,497 params frozen)
- Trainable Causal Unidirectional GRU + Classifier head (39,361 params)
- Candidate sequence lengths ablated on validation: L ∈ {1, 4, 8, 12}
- Dynamic Negative Subsampling (10:1 ratio, seed = base_seed + epoch, base_seed = 42)
- Binary Focal Loss (gamma = 2.0, alpha = 0.25)
- Primary label: label_50pct_overlap (Strategy B >= 50% overlap)
- Strict patient isolation (16 Train / 4 Val / 4 Test)
- Validation-driven checkpoint selection (Peak Validation AUPRC)
- Selection & Freeze of best configuration strictly on validation data
"""

import os
import sys
import time
import json
import hashlib
import platform
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from neuroaegis.models.baselines.model_c import CNN_GNN_GRU
from research.experiments.model_c.sequence_dataset import SequenceBuilder, SequenceNegativeSampler
from research.experiments.model_c.embedding_cache import EmbeddingCacheManager
from research.experiments.imbalance.patient_splitter import PatientDataSplitter
from research.experiments.imbalance.focal_loss import BinaryFocalLossWithLogits, logits_to_probabilities
from neuroaegis.evaluation.window_metrics import SeizureEvaluationMetrics

# Configuration & Paths
MANIFEST_DIR = os.path.join(BASE_DIR, "data/manifests")
WINDOW_INDEX_PATH = os.path.join(MANIFEST_DIR, "chbmit_window_index.csv")
EVENTS_PATH = os.path.join(MANIFEST_DIR, "chbmit_seizure_events.csv")
MANIFEST_PATH = os.path.join(MANIFEST_DIR, "chbmit_manifest.csv")
FROZEN_BACKBONE_PATH = os.path.join(BASE_DIR, "artifacts/checkpoints/frozen_cnn_gnn.pt")
FROZEN_GRAPH_CFG_PATH = os.path.join(BASE_DIR, "research/experiments/gnn/frozen_graph_config.json")
FROZEN_GRAPH_ADJ_PATH = os.path.join(BASE_DIR, "research/experiments/gnn/frozen_graph_adjacency.csv")

PHASE_4B_DIR = os.path.join(BASE_DIR, "research/experiments/model_c")
EXP_BASE_DIR = os.path.join(PHASE_4B_DIR, "experiments")
CACHE_DIR = os.path.join(PHASE_4B_DIR, "embeddings_cache")
LIVE_STATUS_PATH = os.path.join(PHASE_4B_DIR, "live_status.json")
COMPARISON_CSV_PATH = os.path.join(PHASE_4B_DIR, "validation_sequence_comparison.csv")
FROZEN_GRU_CONFIG_PATH = os.path.join(PHASE_4B_DIR, "frozen_gru_config.json")
FROZEN_MODEL_PATH = os.path.join(PHASE_4B_DIR, "frozen_cnn_gnn_gru.pt")

LABEL_COLUMN = "label_50pct_overlap"
SAMPLING_RATIO = 10.0
BASE_SEED = 42
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 3
CANDIDATE_L_VALUES = [1, 4, 8, 12]


def get_file_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def update_live_status(
    stage: str,
    completed_steps: List[str],
    in_progress: str,
    queued_steps: List[str],
    start_time: float,
    extra_info: Optional[Dict[str, Any]] = None
):
    elapsed = time.time() - start_time
    status = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "elapsed_seconds": round(elapsed, 1),
        "completed": completed_steps,
        "in_progress": in_progress,
        "queued": queued_steps,
        "extra_info": extra_info or {}
    }
    with open(LIVE_STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)


def compute_detection_delay_details(
    window_df: pd.DataFrame,
    events_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, int, int, List[Dict[str, Any]]]:
    df = window_df.copy()
    df["pred_prob"] = y_prob
    df["pred_label"] = (df["pred_prob"] >= threshold).astype(int)
    
    delays = []
    event_details = []
    recs = set(window_df["recording_id"].unique())
    sub_events = events_df[events_df["recording_id"].isin(recs)]
    total_events = len(sub_events)
    
    for _, ev in sub_events.iterrows():
        rec_id = ev["recording_id"]
        sz_id = ev.get("seizure_id", f"{rec_id}_sz")
        pat_id = ev.get("patient_id", rec_id.split("_")[0])
        s_start = ev["start_sec"]
        s_end = ev["end_sec"]
        s_dur = ev["duration_sec"]
        
        rec_w = df[df["recording_id"] == rec_id]
        ov_mask = (rec_w["window_end_sec"] > s_start) & (rec_w["window_start_sec"] < s_end)
        ov_w = rec_w[ov_mask]
        
        det_w = ov_w[ov_w["pred_label"] == 1]
        is_detected = len(det_w) > 0
        if is_detected:
            first_alarm = det_w["window_end_sec"].min()
            delay = max(0.0, float(first_alarm - s_start))
            delays.append(delay)
        else:
            first_alarm = None
            delay = None
            
        event_details.append({
            "patient_id": pat_id,
            "recording_id": rec_id,
            "seizure_id": sz_id,
            "start_sec": s_start,
            "end_sec": s_end,
            "duration_sec": s_dur,
            "detected": is_detected,
            "first_alarm_sec": first_alarm,
            "detection_delay_sec": delay,
            "overlapping_positive_windows": len(det_w)
        })
        
    detected_count = len(delays)
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    return round(mean_delay, 2), detected_count, total_events, event_details


def compute_patient_breakdown(
    window_df: pd.DataFrame,
    events_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    df = window_df.copy()
    df["pred_prob"] = y_prob
    df["pred_label"] = (df["pred_prob"] >= threshold).astype(int)
    
    results = []
    for pat_id, pat_w in df.groupby("patient_id"):
        y_t = pat_w[LABEL_COLUMN].values
        y_p = pat_w["pred_prob"].values
        pat_m = SeizureEvaluationMetrics.compute_window_metrics(y_t, y_p, threshold=threshold, stride_sec=2.5)
        
        pat_events = events_df[events_df["patient_id"] == pat_id]
        total_ev = len(pat_events)
        det_ev = 0
        delays = []
        for _, ev in pat_events.iterrows():
            r_id = ev["recording_id"]
            s_s = ev["start_sec"]
            s_e = ev["end_sec"]
            ev_w = pat_w[(pat_w["recording_id"] == r_id) & (pat_w["window_end_sec"] > s_s) & (pat_w["window_start_sec"] < s_e)]
            pos_w = ev_w[ev_w["pred_label"] == 1]
            if len(pos_w) > 0:
                det_ev += 1
                alarm_t = pos_w["window_end_sec"].min()
                delays.append(max(0.0, alarm_t - s_s))
                
        results.append({
            "patient_id": pat_id,
            "total_windows": len(pat_w),
            "recording_hours": round(len(pat_w) * 2.5 / 3600.0, 2),
            "num_seizures": total_ev,
            "detected_seizures": det_ev,
            "missed_seizures": total_ev - det_ev,
            "event_sensitivity": round(det_ev / total_ev, 4) if total_ev > 0 else 0.0,
            "window_sensitivity": pat_m["sensitivity"],
            "window_specificity": pat_m["specificity"],
            "false_alarms_count": pat_m["false_positives"],
            "false_alarms_per_day": pat_m["false_alarms_per_24h"],
            "mean_detection_delay_sec": round(float(np.mean(delays)), 2) if delays else None
        })
    return results


def run_phase_4b_training():
    print("=" * 80)
    print("NEUROAEGIS PHASE 4B: CNN + GNN + CAUSAL GRU TEMPORAL TRAINING")
    print("=" * 80)
    start_time = time.time()
    
    completed_steps = []
    queued_steps = [
        "Partition Verification & Leakage Checks",
        "Load Validation & Training Embeddings Cache",
        "Train & Validate L=1 (Single Window Ablation)",
        "Train & Validate L=4 (12.5s Temporal Span)",
        "Train & Validate L=8 (22.5s Temporal Span)",
        "Train & Validate L=12 (32.5s Temporal Span)",
        "Validation-Only Best Model Selection & Freeze"
    ]
    
    update_live_status("INITIALIZATION", completed_steps, "Hardware & Partition Setup", queued_steps, start_time)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Hardware] Execution device: {device}")
    
    # 1. Partitions & Leakage Checks
    print("\n[Step 1/7] Verifying dataset partitions & zero-leakage invariants...")
    splitter = PatientDataSplitter(label_column=LABEL_COLUMN)
    train_df, val_df, test_df = splitter.get_splits()
    events_df = pd.read_csv(EVENTS_PATH)
    manifest_df = pd.read_csv(MANIFEST_PATH)
    
    # Invariant assertions
    assert len(set(splitter.train_patients) & set(splitter.val_patients)) == 0, "Patient leakage train/val!"
    assert len(set(splitter.train_patients) & set(splitter.test_patients)) == 0, "Patient leakage train/test!"
    assert len(set(splitter.val_patients) & set(splitter.test_patients)) == 0, "Patient leakage val/test!"
    assert len(set(train_df["recording_id"]) & set(val_df["recording_id"])) == 0, "Recording leakage train/val!"
    assert len(set(train_df["recording_id"]) & set(test_df["recording_id"])) == 0, "Recording leakage train/test!"
    assert len(set(val_df["recording_id"]) & set(test_df["recording_id"])) == 0, "Recording leakage val/test!"
    
    idx_hash = get_file_sha256(WINDOW_INDEX_PATH)
    print(f"Master window index SHA256: {idx_hash}")
    assert idx_hash == "f76dddb19fa2c24b3af24b989168b12465512b650c5e6d84bb0125a6d666302c", "Index hash mismatch!"
    
    completed_steps.append("Partition Verification & Leakage Checks (PASS)")
    queued_steps.remove("Partition Verification & Leakage Checks")
    
    # 2. Embedding Cache Manager
    print("\n[Step 2/7] Initializing EmbeddingCacheManager and loading embeddings cache...")
    update_live_status("CACHE_LOADING", completed_steps, "Loading Unified Embeddings Cache", queued_steps, start_time)
    cache_mgr = EmbeddingCacheManager(cache_dir=CACHE_DIR, device=device)
    
    # Load Validation Embeddings
    val_unified_path = os.path.join(CACHE_DIR, "val_embeddings_unified.npy")
    if os.path.exists(val_unified_path) and os.path.getsize(val_unified_path) > 1024:
        print("Loading precomputed unified validation embeddings from disk...")
        val_embs = np.load(val_unified_path)
    else:
        print("Computing validation embeddings...")
        val_embs = cache_mgr.get_split_embeddings(val_df, split_name="val")
    print(f"Validation embeddings ready: shape {val_embs.shape} ({val_embs.nbytes / 1024 / 1024:.1f} MB)")
    
    # Load Training Embeddings
    train_unified_path = os.path.join(CACHE_DIR, "train_embeddings_unified.npy")
    if os.path.exists(train_unified_path) and os.path.getsize(train_unified_path) > 1024:
        print("Loading precomputed unified training embeddings from disk...")
        train_embs = np.load(train_unified_path)
    else:
        print("Computing training embeddings...")
        train_embs = cache_mgr.get_split_embeddings(
            train_df,
            split_name="train",
            progress_callback=lambda cur, tot, r: print(f"  [Train Cache] {cur}/{tot} recordings ({r})...", flush=True) if cur % 20 == 0 or cur == tot else None
        )
    print(f"Training embeddings ready: shape {train_embs.shape} ({train_embs.nbytes / 1024 / 1024:.1f} MB)")
    
    # Create padded arrays where index -1 accesses the zero vector
    zero_pad = np.zeros((1, 128), dtype=np.float32)
    val_embs_padded = np.vstack([val_embs, zero_pad])
    train_embs_padded = np.vstack([train_embs, zero_pad])
    
    completed_steps.append(f"Load Embeddings Cache ({len(val_embs):,} val / {len(train_embs):,} train)")
    queued_steps.remove("Load Validation & Training Embeddings Cache")
    
    # 3. Ablation Training for L in {1, 4, 8, 12}
    builder = SequenceBuilder(label_column=LABEL_COLUMN)
    val_manifest_duration_hours = manifest_df[manifest_df["patient_id"].isin(splitter.val_patients)]["recording_duration_sec"].sum() / 3600.0
    
    all_experiments_summary = []
    
    for seq_len in CANDIDATE_L_VALUES:
        exp_id = f"PHASE4B_GRU_L{seq_len:02d}"
        exp_dir = os.path.join(EXP_BASE_DIR, f"L{seq_len}")
        os.makedirs(exp_dir, exist_ok=True)
        
        step_name = f"Train & Validate L={seq_len} ({builder.compute_temporal_span(seq_len):.1f}s Temporal Span)"
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {exp_id} (Sequence Length L={seq_len}, Temporal Span={builder.compute_temporal_span(seq_len):.1f}s)")
        print("=" * 80)
        
        update_live_status("TRAINING", completed_steps, f"Running {exp_id}", queued_steps, start_time)
        
        # Build sequence metadata
        print(f"Building sequence metadata for L={seq_len}...")
        t_b0 = time.time()
        val_seq_df = builder.build_sequences(val_df, seq_len=seq_len)
        train_seq_df = builder.build_sequences(train_df, seq_len=seq_len)
        t_b1 = time.time()
        print(f"Sequence metadata built in {t_b1 - t_b0:.2f}s (Train: {len(train_seq_df):,}, Val: {len(val_seq_df):,})")
        
        train_seq_sampler = SequenceNegativeSampler(
            train_seq_df,
            ratio=SAMPLING_RATIO,
            base_seed=BASE_SEED,
            shuffle=True
        )
        
        # Pre-extract index matrices
        val_indices_matrix = np.array(val_seq_df["window_indices"].tolist(), dtype=np.int32)
        train_indices_matrix = np.array(train_seq_df["window_indices"].tolist(), dtype=np.int32)
        val_labels = val_seq_df["target_label"].values.astype(np.float32)
        train_labels = train_seq_df["target_label"].values.astype(np.float32)
        
        # Fast vectorized validation tensor construction
        print(f"Constructing vectorized validation tensor for L={seq_len} ({len(val_indices_matrix):,} sequences)...")
        t_v0 = time.time()
        val_seq_tensor = val_embs_padded[val_indices_matrix]
        t_val_x = torch.from_numpy(val_seq_tensor)
        t_v1 = time.time()
        print(f"Validation tensor ready in {t_v1 - t_v0:.2f}s (shape: {t_val_x.shape})")
        
        # Initialize fresh model
        torch.manual_seed(BASE_SEED)
        np.random.seed(BASE_SEED)
        model = CNN_GNN_GRU(frozen_backbone_path=FROZEN_BACKBONE_PATH, device=device)
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
        criterion = BinaryFocalLossWithLogits(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, reduction="mean")
        
        best_val_auprc = -1.0
        best_epoch = -1
        best_val_metrics = None
        best_val_probs = None
        history = []
        
        # Training loop across epochs
        for epoch in range(NUM_EPOCHS):
            t_ep_start = time.time()
            train_seq_sampler.set_epoch(epoch)
            epoch_seq_indices = train_seq_sampler.current_epoch_indices
            
            # Fast vectorized training batch tensor construction
            sub_indices_matrix = train_indices_matrix[epoch_seq_indices]
            train_x_np = train_embs_padded[sub_indices_matrix]
            train_y_np = train_labels[epoch_seq_indices]
            
            train_dataset = TensorDataset(torch.from_numpy(train_x_np), torch.from_numpy(train_y_np))
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0, pin_memory=False)
            
            # Model train pass
            model.train()
            train_loss = 0.0
            train_preds = []
            train_trues = []
            
            t_tr0 = time.time()
            for bx, by in train_loader:
                bx = bx.to(device)
                by = by.to(device)
                
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * len(by)
                probs = logits_to_probabilities(logits).detach().cpu().numpy()
                train_preds.append(probs)
                train_trues.append(by.cpu().numpy())
                del bx, by, logits, loss
                
            scheduler.step()
            train_loss /= len(train_dataset)
            t_tr1 = time.time()
            
            y_tr_pred = np.concatenate(train_preds)
            y_tr_true = np.concatenate(train_trues)
            tr_metrics = SeizureEvaluationMetrics.compute_window_metrics(y_tr_true, y_tr_pred, threshold=0.5)
            del train_dataset, train_loader, train_x_np, train_preds, train_trues, y_tr_pred, y_tr_true
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Full Validation Pass
            model.eval()
            val_preds = []
            t_val0 = time.time()
            with torch.no_grad():
                for b_i in range(0, len(t_val_x), 1024):
                    b_chunk = t_val_x[b_i:b_i+1024].to(device)
                    logits = model(b_chunk)
                    p = logits_to_probabilities(logits).cpu().numpy()
                    val_preds.append(p)
                    del b_chunk, logits
            val_probs = np.concatenate(val_preds)
            t_val1 = time.time()
            del val_preds
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            val_metrics = SeizureEvaluationMetrics.compute_window_metrics(
                y_true=val_labels,
                y_prob=val_probs,
                threshold=0.5,
                total_duration_hours=val_manifest_duration_hours,
                stride_sec=2.5
            )

            
            # Event metrics
            det_delay, det_ev, tot_ev, _ = compute_detection_delay_details(
                window_df=val_df,
                events_df=events_df,
                y_prob=val_probs,
                threshold=0.5
            )
            val_metrics["event_level_sensitivity"] = round(det_ev / tot_ev, 4) if tot_ev > 0 else 0.0
            val_metrics["mean_detection_delay_sec"] = det_delay
            val_metrics["detected_seizures"] = det_ev
            val_metrics["total_seizures"] = tot_ev
            
            # Validation loss
            t_v_true = torch.from_numpy(val_labels).float()
            t_v_p = torch.from_numpy(val_probs).float()
            eps = 1e-7
            t_v_logit = torch.log(torch.clamp(t_v_p, eps, 1.0 - eps) / (1.0 - torch.clamp(t_v_p, eps, 1.0 - eps)))
            val_loss = criterion(t_v_logit, t_v_true).item()
            
            val_auprc = val_metrics["auprc"] or 0.0
            val_auroc = val_metrics["auroc"] or 0.0
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} [{exp_id}] (Train: {t_tr1-t_tr0:.1f}s, Val: {t_val1-t_val0:.1f}s):")
            print(f"  Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            print(f"  Val AUPRC:  {val_auprc:.5f} (Primary Selection Metric) | Val AUROC: {val_auroc:.5f}")
            print(f"  Val Sens:   {val_metrics['sensitivity']*100:.2f}% | Val Spec: {val_metrics['specificity']*100:.2f}%")
            print(f"  Val Events: {det_ev}/{tot_ev} ({val_metrics['event_level_sensitivity']*100:.1f}%) | Delay: {det_delay}s | FA/24h: {val_metrics['false_alarms_per_24h']:.2f}")
            
            ep_rec = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "train_accuracy": tr_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_precision": tr_metrics["precision"],
                "val_precision": val_metrics["precision"],
                "train_recall": tr_metrics["sensitivity"],
                "val_recall": val_metrics["sensitivity"],
                "train_f1": tr_metrics["f1_score"],
                "val_f1": val_metrics["f1_score"],
                "train_auroc": tr_metrics["auroc"],
                "val_auroc": val_auroc,
                "train_auprc": tr_metrics["auprc"],
                "val_auprc": val_auprc,
                "val_event_sensitivity": val_metrics["event_level_sensitivity"],
                "val_false_alarms_per_24h": val_metrics["false_alarms_per_24h"],
                "val_detection_delay_sec": det_delay,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_duration_sec": round(time.time() - t_ep_start, 1)
            }
            history.append(ep_rec)
            
            # Checkpoint selection: Peak Validation AUPRC
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_epoch = epoch + 1
                best_val_metrics = val_metrics
                best_val_probs = val_probs
                
                ckpt_save = {
                    "experiment_id": exp_id,
                    "seq_len": seq_len,
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auprc": best_val_auprc,
                    "val_metrics": best_val_metrics
                }
                torch.save(ckpt_save, os.path.join(exp_dir, "best_model.pt"))
                
        # Save experiment outputs
        pd.DataFrame(history).to_csv(os.path.join(exp_dir, "training_history.csv"), index=False)
        with open(os.path.join(exp_dir, "val_metrics.json"), "w") as f:
            json.dump(best_val_metrics, f, indent=2)
        np.savez_compressed(os.path.join(exp_dir, "val_predictions.npz"), y_true=val_labels, y_prob=best_val_probs)
        
        # Patient breakdown
        pat_breakdown = compute_patient_breakdown(val_df, events_df, best_val_probs, threshold=0.5)
        with open(os.path.join(exp_dir, "val_patient_metrics.json"), "w") as f:
            json.dump(pat_breakdown, f, indent=2)
            
        all_experiments_summary.append({
            "experiment_id": exp_id,
            "seq_len": seq_len,
            "temporal_span_sec": builder.compute_temporal_span(seq_len),
            "best_epoch": best_epoch,
            "val_auprc": best_val_metrics["auprc"],
            "val_auroc": best_val_metrics["auroc"],
            "val_sensitivity": best_val_metrics["sensitivity"],
            "val_specificity": best_val_metrics["specificity"],
            "val_precision": best_val_metrics["precision"],
            "val_f1": best_val_metrics["f1_score"],
            "val_balanced_accuracy": best_val_metrics["balanced_accuracy"],
            "val_event_sensitivity": best_val_metrics["event_level_sensitivity"],
            "val_detected_events": best_val_metrics["detected_seizures"],
            "val_total_events": best_val_metrics["total_seizures"],
            "val_detection_delay_sec": best_val_metrics["mean_detection_delay_sec"],
            "val_false_alarms": best_val_metrics["false_positives"],
            "val_fa_per_24h": best_val_metrics["false_alarms_per_24h"]
        })
        
        completed_steps.append(step_name)
        if step_name in queued_steps:
            queued_steps.remove(step_name)
        elif len(queued_steps) > 0:
            queued_steps.pop(0)
            
        # Clean up candidate model and validation tensors
        del model, optimizer, scheduler, t_val_x, val_seq_tensor, val_indices_matrix, train_indices_matrix
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        
    # 4. Validation Model Comparison & Selection
    print("\n" + "=" * 80)
    print("VALIDATION COMPARISON & BEST MODEL SELECTION")
    print("=" * 80)
    
    comp_df = pd.DataFrame(all_experiments_summary)
    comp_df.to_csv(COMPARISON_CSV_PATH, index=False)
    print(comp_df[["experiment_id", "seq_len", "val_auprc", "val_auroc", "val_event_sensitivity", "val_fa_per_24h", "val_detection_delay_sec"]].to_string(index=False))
    
    # Primary selection rule: highest Validation AUPRC
    best_candidate = comp_df.sort_values(by=["val_auprc", "val_event_sensitivity"], ascending=[False, False]).iloc[0]
    best_seq_len = int(best_candidate["seq_len"])
    best_exp_id = best_candidate["experiment_id"]
    
    print(f"\n>>> SELECTED CONFIGURATION: {best_exp_id} (L={best_seq_len}) with Val AUPRC = {best_candidate['val_auprc']:.5f}")
    
    # Copy best model checkpoint
    best_ckpt_path = os.path.join(EXP_BASE_DIR, f"L{best_seq_len}", "best_model.pt")
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    torch.save(best_ckpt, FROZEN_MODEL_PATH)
    
    # Export frozen configuration JSON
    frozen_gru_cfg = {
        "phase": "Phase 4B",
        "model_architecture": "CNN_GNN_GRU (Causal)",
        "selected_experiment_id": best_exp_id,
        "selected_sequence_length": best_seq_len,
        "sequence_duration_sec": builder.compute_temporal_span(best_seq_len),
        "window_duration_sec": 5.0,
        "window_stride_sec": 2.5,
        "gru_direction": "unidirectional",
        "gru_input_dim": 128,
        "gru_hidden_dim": 64,
        "gru_layers": 1,
        "gru_dropout": 0.0,
        "classifier_architecture": "Linear(64, 32) -> ReLU -> Dropout(0.30) -> Linear(32, 1)",
        "trainable_parameters": 39361,
        "frozen_backbone_parameters": 52497,
        "total_parameters": 91858,
        "loss_function": "BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25)",
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "scheduler": "CosineAnnealingLR(T_max=3, eta_min=1e-5)",
        "negative_sampling_ratio": 10.0,
        "primary_label_rule": "label_50pct_overlap (Strategy B)",
        "frozen_graph_threshold": 0.30,
        "frozen_graph_config_path": FROZEN_GRAPH_CFG_PATH,
        "frozen_graph_adjacency_path": FROZEN_GRAPH_ADJ_PATH,
        "frozen_backbone_checkpoint": FROZEN_BACKBONE_PATH,
        "frozen_model_checkpoint": FROZEN_MODEL_PATH,
        "selection_split": "validation",
        "selection_metric_primary": f"Validation AUPRC ({best_candidate['val_auprc']:.5f})",
        "selection_metric_secondary": f"Validation Event Sensitivity ({best_candidate['val_event_sensitivity']*100:.1f}%), FA/24h ({best_candidate['val_fa_per_24h']:.2f})",
        "selection_rationale": "Selected strictly on validation data based on highest Validation AUPRC among candidate sequence lengths L in {1, 4, 8, 12}.",
        "best_epoch": int(best_candidate["best_epoch"]),
        "validation_metrics": {
            "auprc": float(best_candidate["val_auprc"]),
            "auroc": float(best_candidate["val_auroc"]),
            "event_sensitivity": float(best_candidate["val_event_sensitivity"]),
            "false_alarms_per_24h": float(best_candidate["val_fa_per_24h"]),
            "detection_delay_sec": float(best_candidate["val_detection_delay_sec"]) if not pd.isna(best_candidate["val_detection_delay_sec"]) else None
        },
        "git_commit": "17943cdaccfa1d6857f787b91e53b223dbbb8616",
        "freeze_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    with open(FROZEN_GRU_CONFIG_PATH, "w") as f:
        json.dump(frozen_gru_cfg, f, indent=2)
    print(f"Frozen configuration exported to: {FROZEN_GRU_CONFIG_PATH}")
    
    completed_steps.append("Validation-Only Best Model Selection & Freeze (DONE)")
    if "Validation-Only Best Model Selection & Freeze" in queued_steps:
        queued_steps.remove("Validation-Only Best Model Selection & Freeze")
    elif len(queued_steps) > 0:
        queued_steps.clear()
    
    update_live_status("COMPLETE", completed_steps, "All 4 Validation Ablations Trained & Frozen", queued_steps, start_time, extra_info={"selected_model": best_exp_id})
    print(f"\nPhase 4B Training & Validation completed in {time.time() - start_time:.1f}s.")
    return frozen_gru_cfg


if __name__ == "__main__":
    run_phase_4b_training()
