#!/usr/bin/env python3
"""
extract_multichannel.py
───────────────────────
Downloads CHB-MIT EDF files from PhysioNet and extracts 57 features per channel
for ALL available channels (typically 23), producing a multi-channel parquet file
for the attention pooling model.

Also generates Bonn features.csv from the Bonn University dataset if raw data
is available, or downloads it from the UCI/Andrzejak mirror.

Output:
  - data/chbmit_multichannel.parquet  (23 channels × 57 features per window)
  - data/bonn_features.csv            (single-channel 57 features)

Usage:
  python scripts/extract_multichannel.py [--patients chb01 chb02 ...] [--bonn]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# Add apps/api to path for feature extraction
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
from app.services.features import extract_features, extract_features_multichannel

# ── Constants ────────────────────────────────────────────────────────────────
PHYSIONET_URL = "https://physionet.org/files/chbmit/1.0.0"
BONN_URL = "https://www.ukbonn.de/site/assets/files/22398"  # Andrzejak 2001

FS_CHBMIT = 256.0
FS_BONN = 173.61

# 60-second windows for CHB-MIT (matching V2 predict endpoint)
WINDOW_SEC_CHBMIT = 60.0
SAMPLES_PER_WINDOW_CHBMIT = int(WINDOW_SEC_CHBMIT * FS_CHBMIT)  # 15360

# Bonn segments are 23.6 seconds (4097 samples) — use as-is
BONN_SEGMENT_SAMPLES = 4097

ALL_CHBMIT_PATIENTS = [f"chb{i:02d}" for i in range(1, 25)]
DEFAULT_PATIENTS = ALL_CHBMIT_PATIENTS

DATA_DIR = REPO_ROOT / "data"
CHBMIT_DIR = DATA_DIR / "chbmit_edf"
OUTPUT_MULTICHANNEL = DATA_DIR / "chbmit_multichannel.parquet"
OUTPUT_BONN = DATA_DIR / "bonn_features.csv"


# ── Utilities ────────────────────────────────────────────────────────────────
def download_file(url: str, out_path: Path, max_retries: int = 3) -> None:
    """Download a file with retry logic and large 1MB buffer."""
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  ↓ {out_path.name} (attempt {attempt}/{max_retries})")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(out_path, "wb") as f:
                while True:
                    chunk = resp.read(1048576)  # 1MB buffer for fast streaming
                    if not chunk:
                        break
                    f.write(chunk)
            return
        except Exception as e:
            if out_path.exists():
                out_path.unlink()
            print(f"    ✗ {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")


def parse_summary(summary_path: Path) -> dict:
    """Parse CHB-MIT summary file → {edf_filename: [(start_sec, end_sec), ...]}."""
    seizures: dict[str, list] = {}
    with open(summary_path) as f:
        lines = f.read().splitlines()

    current_file = None
    for i, line in enumerate(lines):
        m = re.search(r"File Name:\s*(chb\d+[_\w]*\.edf)", line)
        if m:
            current_file = m.group(1)
            seizures.setdefault(current_file, [])

        m_start = re.search(r"Seizure\s+(?:\d+\s+)?Start Time:\s*(\d+)", line)
        if m_start and current_file is not None:
            start_sec = int(m_start.group(1))
            for j in range(i + 1, min(i + 5, len(lines))):
                m_end = re.search(r"Seizure\s+(?:\d+\s+)?End Time:\s*(\d+)", lines[j])
                if m_end:
                    seizures[current_file].append((start_sec, int(m_end.group(1))))
                    break
    return seizures


# ── CHB-MIT Multi-Channel Extraction ─────────────────────────────────────────
def extract_chbmit_multichannel(patients: list[str], output_path: Path = OUTPUT_MULTICHANNEL, delete_edf: bool = True) -> None:
    """Download and extract multi-channel features for specified CHB-MIT patients."""
    print("=" * 70)
    print("CHB-MIT Multi-Channel Feature Extraction")
    print("=" * 70)

    if output_path.exists():
        print(f"\n⚠  Output already exists: {output_path}")
        print("   Delete it to re-extract.\n")
        return

    # Download global metadata
    for meta in ["SUBJECT-INFO", "RECORDS", "RECORDS-WITH-SEIZURES"]:
        download_file(f"{PHYSIONET_URL}/{meta}", DATA_DIR / "chbmit_subset" / meta)

    records_path = DATA_DIR / "chbmit_subset" / "RECORDS"
    with open(records_path) as f:
        all_records = f.read().splitlines()

    all_rows: list[dict] = []
    total_windows = 0
    total_seizure_windows = 0
    patient_stats: list[dict] = []

    for patient in patients:
        print(f"\n{'─'*50}")
        print(f"Patient: {patient}")
        print(f"{'─'*50}")

        patient_edf_dir = CHBMIT_DIR / patient
        patient_edf_dir.mkdir(parents=True, exist_ok=True)

        # Download + parse summary
        summary_url = f"{PHYSIONET_URL}/{patient}/{patient}-summary.txt"
        summary_path = DATA_DIR / "chbmit_subset" / patient / f"{patient}-summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            download_file(summary_url, summary_path)
            seizure_dict = parse_summary(summary_path)
        except Exception as e:
            print(f"  ⚠ Summary file download failed for {patient}: {e}. Skipping patient.")
            patient_stats.append({"patient_id": patient, "status": "skipped", "reason": str(e), "windows": 0, "seizures": 0})
            continue

        # Get patient records
        patient_records = [r for r in all_records if r.startswith(f"{patient}/")]
        if not patient_records:
            print(f"  ⚠ No records found in RECORDS file for {patient}. Skipping.")
            patient_stats.append({"patient_id": patient, "status": "no_records", "reason": "Not in RECORDS", "windows": 0, "seizures": 0})
            continue

        patient_windows = 0
        patient_seizures = 0

        # For efficiency: process files with seizures + first background file
        processed_background = False
        for record in patient_records:
            edf_name = record.split("/")[1]
            has_seizure = len(seizure_dict.get(edf_name, [])) > 0

            if not has_seizure and processed_background:
                continue
            if not has_seizure:
                processed_background = True

            edf_url = f"{PHYSIONET_URL}/{record}"
            edf_path = patient_edf_dir / edf_name

            try:
                download_file(edf_url, edf_path)
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
                data = raw.get_data()  # (n_channels, n_samples)
                ch_names = raw.ch_names
                n_channels, n_total_samples = data.shape

                # Skip non-EEG channels (e.g., ECG, VNS)
                eeg_mask = []
                eeg_names = []
                for ci, name in enumerate(ch_names):
                    upper = name.upper().replace(" ", "").replace("-", "")
                    if any(x in upper for x in ["ECG", "VNS", "EMG", "EOG", "EKG", "STI"]):
                        continue
                    eeg_mask.append(ci)
                    eeg_names.append(f"Ch{len(eeg_names)}")
                
                if not eeg_mask:
                    print(f"  ⚠ {edf_name}: no EEG channels found, skipping")
                    continue

                eeg_data = data[eeg_mask, :]  # (n_eeg_channels, n_samples)
                n_eeg_channels = len(eeg_mask)

                sz_times = seizure_dict.get(edf_name, [])
                n_windows = n_total_samples // SAMPLES_PER_WINDOW_CHBMIT

                for w in range(n_windows):
                    start_idx = w * SAMPLES_PER_WINDOW_CHBMIT
                    end_idx = start_idx + SAMPLES_PER_WINDOW_CHBMIT
                    window_start_sec = start_idx / FS_CHBMIT
                    window_end_sec = end_idx / FS_CHBMIT

                    # Label: 1 if window overlaps any seizure period
                    target = 0
                    for sz_start, sz_end in sz_times:
                        if window_start_sec <= sz_end and window_end_sec >= sz_start:
                            target = 1
                            break

                    window_data = eeg_data[:, start_idx:end_idx]

                    # Extract features for ALL channels
                    features = extract_features_multichannel(
                        window_data,
                        channel_names=eeg_names,
                        fs=FS_CHBMIT,
                        wavelet="db4",
                        level=5,
                    )

                    features["target"] = target
                    features["patient_id"] = patient
                    features["record"] = edf_name
                    features["window_idx"] = w
                    features["n_channels"] = n_eeg_channels
                    all_rows.append(features)
                    patient_windows += 1
                    patient_seizures += target
                    total_windows += 1
                    total_seizure_windows += target

                print(f"  ✓ {edf_name}: {n_windows} windows, {n_eeg_channels} channels, "
                      f"{len(sz_times)} seizure(s)")

            except Exception as e:
                print(f"  ✗ {edf_name}: {e}")
            finally:
                if delete_edf and edf_path.exists():
                    edf_path.unlink()

        patient_stats.append({
            "patient_id": patient,
            "status": "success" if patient_windows > 0 else "no_data",
            "reason": "OK" if patient_windows > 0 else "Extraction failed",
            "windows": patient_windows,
            "seizures": patient_seizures
        })

    # Save
    if all_rows:
        df = pd.DataFrame(all_rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"\n{'='*70}")
        print(f"✓ Saved: {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Total Windows: {total_windows} ({total_seizure_windows} seizure windows)")
        print(f"  Patients processed: {len(patient_stats)}")
        print(f"{'='*70}")

        # Log cohort summary
        stats_df = pd.DataFrame(patient_stats)
        print("\nPatient Cohort Summary:")
        print(stats_df.to_string(index=False))
    else:
        print("\n✗ No data extracted!")


# ── Bonn Feature Generation ─────────────────────────────────────────────────
BONN_SETS = {
    "Z": {"url_file": "z.zip", "label": 0, "description": "Healthy, eyes open"},
    "O": {"url_file": "o.zip", "label": 1, "description": "Healthy, eyes closed"},
    "N": {"url_file": "n.zip", "label": 2, "description": "Interictal, opposite hemisphere"},
    "F": {"url_file": "f.zip", "label": 3, "description": "Interictal, seizure focus"},
    "S": {"url_file": "s.zip", "label": 4, "description": "Seizure (ictal)"},
}


def generate_bonn_features() -> None:
    """Generate Bonn features.csv from raw text segments or download them."""
    print("\n" + "=" * 70)
    print("Bonn University Dataset Feature Extraction")
    print("=" * 70)

    if OUTPUT_BONN.exists():
        print(f"\n⚠  Output already exists: {OUTPUT_BONN}")
        print("   Delete it to re-extract.\n")
        return

    bonn_raw_dir = DATA_DIR / "bonn_raw"
    bonn_raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for set_name, info in BONN_SETS.items():
        set_dir = bonn_raw_dir / set_name
        set_dir.mkdir(parents=True, exist_ok=True)

        # Download zip
        zip_url = f"{BONN_URL}/{info['url_file']}"
        zip_path = bonn_raw_dir / info["url_file"]

        try:
            download_file(zip_url, zip_path)

            # Extract zip
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(set_dir)

            # Each txt file is a single-channel 4097-sample segment
            txt_files = sorted(set_dir.glob("*.txt"))
            if not txt_files:
                # Some zips have a subfolder
                txt_files = sorted(set_dir.rglob("*.txt"))

            print(f"  Set {set_name} ({info['description']}): {len(txt_files)} segments")

            for txt_file in txt_files:
                try:
                    signal = np.loadtxt(txt_file)
                    if len(signal) < 100:
                        continue

                    features = extract_features(
                        signal, fs=FS_BONN, wavelet="coif3", level=4
                    )
                    features["label"] = info["label"]
                    features["epoch_id"] = txt_file.stem
                    features["set"] = set_name
                    rows.append(features)
                except Exception as e:
                    print(f"    ✗ {txt_file.name}: {e}")

        except Exception as e:
            print(f"  ✗ Set {set_name}: {e}")
            print(f"    URL attempted: {zip_url}")
            print(f"    If download fails, manually download Bonn data to {set_dir}/")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_BONN, index=False)
        print(f"\n✓ Saved: {OUTPUT_BONN}")
        print(f"  Shape: {df.shape}")
        print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    else:
        print("\n✗ No Bonn data extracted! Creating synthetic placeholder...")
        _create_synthetic_bonn()


def _create_synthetic_bonn() -> None:
    """Create a synthetic Bonn features.csv for testing when download fails."""
    np.random.seed(42)
    rows = []
    for label in range(5):
        for epoch in range(100):
            signal = np.random.randn(BONN_SEGMENT_SAMPLES)
            if label == 4:  # Seizure class — add oscillatory component
                t = np.arange(BONN_SEGMENT_SAMPLES) / FS_BONN
                signal += 3.0 * np.sin(2 * np.pi * 8.0 * t)  # Strong 8 Hz spike

            features = extract_features(signal, fs=FS_BONN, wavelet="coif3", level=4)
            features["label"] = label
            features["epoch_id"] = f"synthetic_{label}_{epoch:03d}"
            features["set"] = chr(ord("A") + label)
            rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_BONN, index=False)
    print(f"  ✓ Synthetic Bonn features saved: {OUTPUT_BONN} ({df.shape})")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract multi-channel EEG features for attention pooling training"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=DEFAULT_PATIENTS,
        help="CHB-MIT patient IDs to process (default: all chb01-chb24)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "chbmit_multichannel_full.parquet"),
        help="Path for saving extracted parquet file",
    )
    parser.add_argument(
        "--bonn",
        action="store_true",
        default=False,
        help="Also generate Bonn features",
    )
    parser.add_argument(
        "--keep-edf",
        action="store_true",
        help="Keep downloaded EDF files (default: delete to save space)",
    )
    args = parser.parse_args()

    extract_chbmit_multichannel(args.patients, output_path=Path(args.output), delete_edf=not args.keep_edf)

    if args.bonn:
        generate_bonn_features()

    print("\n" + "=" * 70)
    print("All done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
