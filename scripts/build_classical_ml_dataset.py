#!/usr/bin/env python3
"""
build_classical_ml_dataset.py
───────────────────────────────
Extracts 57 multi-domain EEG features (22 Time, 14 Frequency, 21 Wavelet DWT)
for every window defined in research/data/manifests/chbmit_window_index.csv
across all 24 CHB-MIT patients using 3D-tensor window batching.

Output:
  data/chbmit_features/chb01.parquet ... chb24.parquet
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
import pywt
from scipy.signal import welch
from scipy.stats import skew, kurtosis, iqr, entropy

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = REPO_ROOT / "research" / "data" / "manifests" / "chbmit_window_index.csv"
DATASET_DIR = REPO_ROOT / "CHB-MIT Dataset"
OUTPUT_DIR = REPO_ROOT / "data" / "chbmit_features"

FS = 256.0
WIN_SIZE = 1280


def extract_features_3d_tensor(wins_3d: np.ndarray, fs: float = 256.0) -> pd.DataFrame:
    """
    Extracts 57 multi-domain features vectorized over (n_wins, n_ch, 1280).
    Returns DataFrame of shape (n_wins, 57).
    """
    means = np.mean(wins_3d, axis=2)
    medians = np.median(wins_3d, axis=2)
    stds = np.std(wins_3d, axis=2)
    vars_ = np.var(wins_3d, axis=2)
    mins = np.min(wins_3d, axis=2)
    maxs = np.max(wins_3d, axis=2)
    ranges = np.ptp(wins_3d, axis=2)
    rms = np.sqrt(np.mean(wins_3d**2, axis=2))
    energies = np.sum(wins_3d**2, axis=2)
    abs_means = np.mean(np.abs(wins_3d), axis=2)
    peaks = np.max(np.abs(wins_3d), axis=2)
    line_lens = np.sum(np.abs(np.diff(wins_3d, axis=2)), axis=2)
    zero_cross = np.sum(np.diff(np.sign(wins_3d), axis=2) != 0, axis=2)
    skews = skew(wins_3d, axis=2)
    kurts = kurtosis(wins_3d, axis=2)
    iqrs = iqr(wins_3d, axis=2)
    
    crest = peaks / (rms + 1e-12)
    shape = rms / (abs_means + 1e-12)
    impulse = peaks / (abs_means + 1e-12)
    clearance = peaks / ((np.mean(np.sqrt(np.abs(wins_3d)), axis=2)**2) + 1e-12)
    
    d1 = np.diff(wins_3d, axis=2)
    d2 = np.diff(d1, axis=2)
    act = vars_
    mob = np.sqrt(np.var(d1, axis=2) / (act + 1e-12))
    comp = np.sqrt(np.var(d2, axis=2) / (np.var(d1, axis=2) + 1e-12)) / (mob + 1e-12)
    
    freqs, psd = welch(wins_3d, fs=fs, nperseg=min(512, WIN_SIZE), axis=2)
    tot_pow = np.trapz(psd, freqs, axis=2) + 1e-12
    
    def bp(l, h):
        idx = (freqs >= l) & (freqs <= h)
        return np.trapz(psd[:, :, idx], freqs[idx], axis=2)
        
    delta = bp(0.5, 4)
    theta = bp(4, 8)
    alpha = bp(8, 13)
    beta = bp(13, 30)
    gamma = bp(30, 45)
    
    dom_freq = freqs[np.argmax(psd, axis=2)]
    psd_norm = psd / np.sum(psd, axis=2, keepdims=True)
    p_safe = np.clip(psd_norm, 1e-15, 1.0)
    spec_ent = -np.sum(psd_norm * np.log(p_safe), axis=2)
    spec_cent = np.sum(freqs * psd, axis=2) / np.sum(psd, axis=2)
    
    coeffs = pywt.wavedec(wins_3d, "db4", level=4, axis=2)
    w_energies = [np.sum(c**2, axis=2) for c in coeffs]
    w_tot = sum(w_energies) + 1e-12
    w_probs = np.stack(w_energies, axis=-1) / w_tot[:, :, None]
    w_p_safe = np.clip(w_probs, 1e-15, 1.0)
    w_ent = -np.sum(w_probs * np.log(w_p_safe), axis=2)
    
    feats = {
        "mean": np.mean(means, axis=1), "median": np.mean(medians, axis=1), "std": np.mean(stds, axis=1), "variance": np.mean(vars_, axis=1),
        "minimum": np.mean(mins, axis=1), "maximum": np.mean(maxs, axis=1), "range": np.mean(ranges, axis=1), "rms": np.mean(rms, axis=1),
        "energy": np.mean(energies, axis=1), "absolute_mean": np.mean(abs_means, axis=1), "line_length": np.mean(line_lens, axis=1),
        "zero_crossings": np.mean(zero_cross, axis=1), "skewness": np.mean(skews, axis=1), "kurtosis": np.mean(kurts, axis=1),
        "iqr": np.mean(iqrs, axis=1), "crest_factor": np.mean(crest, axis=1), "shape_factor": np.mean(shape, axis=1),
        "impulse_factor": np.mean(impulse, axis=1), "clearance_factor": np.mean(clearance, axis=1),
        "hjorth_activity": np.mean(act, axis=1), "hjorth_mobility": np.mean(mob, axis=1), "hjorth_complexity": np.mean(comp, axis=1),
        "delta_power": np.mean(delta, axis=1), "theta_power": np.mean(theta, axis=1), "alpha_power": np.mean(alpha, axis=1),
        "beta_power": np.mean(beta, axis=1), "gamma_power": np.mean(gamma, axis=1),
        "relative_delta": np.mean(delta / tot_pow, axis=1), "relative_theta": np.mean(theta / tot_pow, axis=1),
        "relative_alpha": np.mean(alpha / tot_pow, axis=1), "relative_beta": np.mean(beta / tot_pow, axis=1),
        "relative_gamma": np.mean(gamma / tot_pow, axis=1), "dominant_frequency": np.mean(dom_freq, axis=1),
        "spectral_entropy": np.mean(spec_ent, axis=1), "spectral_centroid": np.mean(spec_cent, axis=1),
        "total_power": np.mean(tot_pow, axis=1), "wavelet_entropy": np.mean(w_ent, axis=1)
    }
    for i, c in enumerate(coeffs):
        feats[f"wavelet_energy_{i}"] = np.mean(w_energies[i], axis=1)
        feats[f"wavelet_relative_energy_{i}"] = np.mean(w_energies[i] / w_tot, axis=1)
        feats[f"wavelet_mean_{i}"] = np.mean(np.mean(c, axis=2), axis=1)
        feats[f"wavelet_std_{i}"] = np.mean(np.std(c, axis=2), axis=1)
        
    return pd.DataFrame(feats)


def process_patient(patient_id: str, pt_df: pd.DataFrame) -> str:
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

    for edf_name in records:
        edf_path = pt_dir / edf_name
        if not edf_path.exists():
            continue

        edf_wins = pt_df[pt_df["edf_filename"] == edf_name].copy()
        if edf_wins.empty:
            continue

        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            data = raw.get_data() # (n_ch, n_samples)
            
            # Extract windows for this EDF
            valid_rows = []
            win_tensors = []
            for _, row in edf_wins.iterrows():
                s_start = int(row["window_start_sample"])
                s_end = int(row["window_end_sample"])
                if s_end <= data.shape[1] and (s_end - s_start) == WIN_SIZE:
                    win_tensors.append(data[:, s_start:s_end])
                    valid_rows.append(row)

            if not win_tensors:
                del raw, data
                gc.collect()
                continue

            wins_3d = np.stack(win_tensors, axis=0)
            feat_df = extract_features_3d_tensor(wins_3d, fs=FS)
            
            meta_df = pd.DataFrame(valid_rows).reset_index(drop=True)
            meta_cols = ["window_id", "patient_id", "edf_filename", "window_start_sec", "window_end_sec", "label_50pct_overlap", "primary_label"]
            merged = pd.concat([meta_df[meta_cols], feat_df], axis=1)
            
            patient_dfs.append(merged)
            extracted_count += len(merged)

            del raw, data, win_tensors, wins_3d, feat_df, meta_df, merged
            gc.collect()

        except Exception as e:
            print(f"  Error processing {edf_name}: {e}")

    if patient_dfs:
        res_df = pd.concat(patient_dfs, ignore_index=True)
        res_df.to_parquet(out_file, index=False)
        del patient_dfs, res_df
        gc.collect()
        t1 = time.time()
        return f"✓ Processed {patient_id}: {extracted_count} windows in {t1-t0:.1f}s -> {out_file.name}"
    else:
        return f"✗ Failed to extract windows for {patient_id}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_patients = [f"chb{i:02d}" for i in range(1, 25)]
    
    print("=" * 70)
    print("NeuroAegis - 3D-Tensor Fast Feature Extraction (57 Features)")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Loading Master Index: {INDEX_PATH}")
    print("=" * 70)

    t_start = time.time()
    df_idx = pd.read_csv(INDEX_PATH, low_memory=False)
    print(f"✓ Master Index Loaded ({len(df_idx)} rows) in {time.time() - t_start:.2f}s.")
    
    patient_dfs = {pt: df_idx[df_idx["patient_id"] == pt].copy() for pt in all_patients}
    
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as executor:
        futures = {executor.submit(process_patient, pt, patient_dfs[pt]): pt for pt in all_patients}
        for future in as_completed(futures):
            pt = futures[future]
            try:
                res = future.result()
                print(res)
            except Exception as e:
                print(f"✗ Exception processing {pt}: {e}")

    t_end = time.time()
    print("=" * 70)
    print(f"Feature Extraction Completed in {t_end - t_start:.1f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
