# build_train_cache.py
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from research.phase_3.data_loader import CHBMITDataPipeline
from research.imbalance.patient_splitter import PatientDataSplitter
from research.imbalance.dynamic_sampler import DynamicNegativeSampler
from neuroaegis.utils.memory import flush_memory, get_peak_rss_mb, get_live_rss_mb

CACHE_DIR = BASE_DIR / "research" / "experiments" / "eeg_specific" / "train_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMN = "label_50pct_overlap"
SAMPLING_RATIO = 10.0
BASE_SEED = 42
NUM_EPOCHS = 3

def main():
    print("=" * 80)
    print("BUILDING TRAINING WINDOW CACHE FOR EXPERIMENT 2")
    print("=" * 80)
    t0 = time.time()

    splitter = PatientDataSplitter(label_column=LABEL_COLUMN)
    train_df, val_df, test_df = splitter.get_splits()
    print(f"Train patients ({len(splitter.train_patients)}): {splitter.train_patients}")

    pipeline = CHBMITDataPipeline(edf_root_dir=str(BASE_DIR / "CHB-MIT Dataset"))
    sampler = DynamicNegativeSampler(
        train_df=train_df,
        ratio=SAMPLING_RATIO,
        base_seed=BASE_SEED,
        label_column=LABEL_COLUMN,
        shuffle=True
    )

    pos_win_path = CACHE_DIR / "pos_windows.npy"
    pos_lbl_path = CACHE_DIR / "pos_labels.npy"

    if pos_win_path.exists() and pos_lbl_path.exists():
        print(f"Positive windows already cached: {pos_win_path}")
    else:
        print(f"\n[1/4] Extracting {len(sampler.pos_indices):,} positive training windows...")
        t_pos0 = time.time()
        X_pos, y_pos = pipeline.load_epoch_windows(
            df=train_df,
            sampled_indices=sampler.pos_indices,
            label_column=LABEL_COLUMN,
            shuffle=False,
            progress_callback=lambda cur, tot: print(f"  Positive extraction: {cur}/{tot} recordings...", flush=True) if cur % 20 == 0 or cur == tot else None
        )
        np.save(pos_win_path, X_pos.numpy().astype(np.float32))
        np.save(pos_lbl_path, y_pos.numpy().astype(np.float32))
        print(f"Cached positive windows in {time.time() - t_pos0:.1f}s ({pos_win_path.stat().st_size / 1024 / 1024:.1f} MB)")
        del X_pos, y_pos
        flush_memory()

    for ep in range(NUM_EPOCHS):
        neg_win_path = CACHE_DIR / f"neg_windows_ep{ep}.npy"
        neg_lbl_path = CACHE_DIR / f"neg_labels_ep{ep}.npy"

        if neg_win_path.exists() and neg_lbl_path.exists():
            print(f"Epoch {ep+1} negative windows already cached: {neg_win_path}")
            continue

        print(f"\n[{ep+2}/4] Extracting Epoch {ep+1} negative windows (seed={BASE_SEED + ep})...")
        sampler.set_epoch(ep)
        t_neg0 = time.time()
        X_neg, y_neg = pipeline.load_epoch_windows(
            df=train_df,
            sampled_indices=sampler.current_sampled_neg_indices,
            label_column=LABEL_COLUMN,
            shuffle=False,
            progress_callback=lambda cur, tot: print(f"  Negatives Ep {ep+1}: {cur}/{tot} recordings...", flush=True) if cur % 25 == 0 or cur == tot else None
        )
        np.save(neg_win_path, X_neg.numpy().astype(np.float32))
        np.save(neg_lbl_path, y_neg.numpy().astype(np.float32))
        print(f"Cached Epoch {ep+1} negatives in {time.time() - t_neg0:.1f}s ({neg_win_path.stat().st_size / 1024 / 1024:.1f} MB)")
        del X_neg, y_neg
        flush_memory()

    print(f"\nAll training window caches ready in {time.time() - t0:.1f}s! Live RSS: {get_live_rss_mb():.1f} MB | Peak RSS: {get_peak_rss_mb():.1f} MB")

if __name__ == "__main__":
    main()
