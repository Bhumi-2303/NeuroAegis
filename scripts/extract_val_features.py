#!/usr/bin/env python3
"""
extract_val_features.py
───────────────────────
Extracts 57 multi-domain EEG features (Time, Frequency, Wavelet)
for missing validation patients (chb06, chb07, chb10).
chb08.parquet already exists.
"""

import os
import sys
import time
import glob
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import mne
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_config"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_classical_ml_dataset import extract_features_3d_tensor

INDEX_PATH = REPO_ROOT / "research" / "data" / "manifests" / "chbmit_window_index.csv"
DATASET_DIR = REPO_ROOT / "CHB-MIT Dataset"
OUTPUT_DIR = REPO_ROOT / "data" / "chbmit_features"

FS = 256.0
WIN_SIZE = 1280
CHUNK_WINS = 1000  # Process max 1000 windows at a time to keep RAM < 150MB per worker


def process_val_patient(patient_id: str, pt_df: pd.DataFrame) -> str:
    import gc
    out_file = OUTPUT_DIR / f"{patient_id}.parquet"
    if out_file.exists():
        return f"✓ {patient_id} already exists ({out_file})"

    if pt_df.empty:
        return f"⚠ No windows found for {patient_id}"

    records = pt_df["edf_filename"].unique()
    pt_dir = DATASET_DIR / patient_id

    patient_dfs = []
    t0 = time.time()
    extracted_count = 0

    print(f"[{patient_id}] Starting extraction across {len(records)} recordings...", flush=True)

    for rec_idx, edf_name in enumerate(records):
        edf_path = pt_dir / edf_name
        if not edf_path.exists():
            continue

        edf_wins = pt_df[pt_df["edf_filename"] == edf_name].copy()
        if edf_wins.empty:
            continue

        t_rec0 = time.time()
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            data = raw.get_data()  # (n_ch, n_samples)

            # Map to canonical 23 if needed
            if data.shape[0] > 23:
                # Keep first 23 standard EEG channels
                data = data[:23]

            valid_rows = []
            win_tensors = []
            for _, row in edf_wins.iterrows():
                s_start = int(row["window_start_sample"])
                s_end = int(row["window_end_sample"])
                if s_end <= data.shape[1] and (s_end - s_start) == WIN_SIZE:
                    win_tensors.append(data[:, s_start:s_end])
                    valid_rows.append(row)

            del raw, data
            gc.collect()

            if not win_tensors:
                continue

            # Process in sub-chunks to keep memory minimal
            n_total = len(win_tensors)
            sub_dfs = []
            meta_cols = ["window_id", "patient_id", "edf_filename", "window_start_sec", "window_end_sec", "label_50pct_overlap", "primary_label"]

            for sub_i in range(0, n_total, CHUNK_WINS):
                sub_tensors = np.stack(win_tensors[sub_i : sub_i + CHUNK_WINS], axis=0)
                sub_feat = extract_features_3d_tensor(sub_tensors, fs=FS)
                sub_meta = pd.DataFrame(valid_rows[sub_i : sub_i + CHUNK_WINS]).reset_index(drop=True)
                sub_dfs.append(pd.concat([sub_meta[meta_cols], sub_feat], axis=1))
                del sub_tensors, sub_feat, sub_meta
                gc.collect()

            edf_merged = pd.concat(sub_dfs, ignore_index=True)
            patient_dfs.append(edf_merged)
            extracted_count += len(edf_merged)

            del win_tensors, valid_rows, sub_dfs, edf_merged
            gc.collect()

            print(f"  [{patient_id}] ({rec_idx+1}/{len(records)}) {edf_name}: {n_total} wins in {time.time()-t_rec0:.1f}s (Total: {extracted_count:,})", flush=True)

        except Exception as e:
            print(f"  [{patient_id}] Error in {edf_name}: {e}", flush=True)

    if patient_dfs:
        res_df = pd.concat(patient_dfs, ignore_index=True)
        res_df.to_parquet(out_file, index=False)
        del patient_dfs, res_df
        gc.collect()
        t1 = time.time()
        return f"✓ Processed {patient_id}: {extracted_count:,} windows in {t1-t0:.1f}s -> {out_file.name}"
    else:
        return f"✗ Failed to extract windows for {patient_id}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    val_targets = ["chb06", "chb07", "chb10"]

    print("=" * 70)
    print(f"Targeting Validation Patients: {val_targets}")
    print(f"Loading Master Index: {INDEX_PATH}")
    print("=" * 70, flush=True)

    t_start = time.time()
    df_idx = pd.read_csv(INDEX_PATH, low_memory=False)
    print(f"✓ Master Index Loaded ({len(df_idx):,} rows) in {time.time() - t_start:.2f}s.", flush=True)

    patient_dfs = {pt: df_idx[df_idx["patient_id"] == pt].copy() for pt in val_targets}

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_val_patient, pt, patient_dfs[pt]): pt for pt in val_targets}
        for future in as_completed(futures):
            pt = futures[future]
            try:
                res = future.result()
                print(f"\n>>> {res}\n", flush=True)
            except Exception as e:
                print(f"✗ Exception in {pt}: {e}", flush=True)

    print("=" * 70)
    print(f"All Validation Features Extracted in {time.time() - t_start:.1f}s!")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
