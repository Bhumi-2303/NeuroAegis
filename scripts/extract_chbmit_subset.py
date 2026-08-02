import os
import re
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
import mne
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# Add apps/api to path so we can import features
sys.path.append(str(Path(__file__).resolve().parent.parent / "apps" / "api"))
from app.services.features import extract_features_multichannel

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
PATIENTS = ["chb01", "chb02", "chb03", "chb04"]
WINDOW_SEC = 23.6
FS = 256  # CHB-MIT is 256Hz
SAMPLES_PER_WINDOW = int(WINDOW_SEC * FS)

DATA_DIR = Path("data/chbmit_subset")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, out_path):
    if not out_path.exists():
        print(f"Downloading {url} to {out_path}")
        urllib.request.urlretrieve(url, out_path)

def parse_summary(summary_path):
    """Parses summary.txt to extract seizure start and end times for each EDF file."""
    seizures = {}
    current_file = None
    with open(summary_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    for line in lines:
        file_match = re.search(r"File Name:\s*(chb\d+_\d+\.edf)", line)
        if file_match:
            current_file = file_match.group(1)
            seizures[current_file] = []
        
        start_match = re.search(r"Seizure\s+(?:\d+\s+)?Start Time:\s*(\d+)", line)
        end_match = re.search(r"Seizure\s+(?:\d+\s+)?End Time:\s*(\d+)", line)
        
        if start_match:
            start_time = int(start_match.group(1))
            # Find the next line for end time
            idx = lines.index(line)
            for j in range(idx + 1, min(idx + 5, len(lines))):
                em = re.search(r"Seizure\s+(?:\d+\s+)?End Time:\s*(\d+)", lines[j])
                if em:
                    end_time = int(em.group(1))
                    seizures[current_file].append((start_time, end_time))
                    break
    return seizures

def main():
    # 1. Download Global Metadata
    for meta_file in ["SUBJECT-INFO", "RECORDS", "RECORDS-WITH-SEIZURES"]:
        download_file(f"{BASE_URL}/{meta_file}", DATA_DIR / meta_file)

    with open(DATA_DIR / "RECORDS", 'r') as f:
        all_records = f.read().splitlines()

    rows = []
    
    # 2. Process patients
    for patient in PATIENTS:
        print(f"--- Processing {patient} ---")
        patient_dir = DATA_DIR / patient
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Download summary
        summary_url = f"{BASE_URL}/{patient}/{patient}-summary.txt"
        summary_path = patient_dir / f"{patient}-summary.txt"
        download_file(summary_url, summary_path)
        
        seizures_dict = parse_summary(summary_path)
        
        # Get EDF files for this patient
        patient_records = [r for r in all_records if r.startswith(f"{patient}/")]
        
        for record in patient_records:
            edf_filename = record.split('/')[1]
            # To save time, we will ONLY process EDF files that have seizures, plus one background file per patient
            # Wait, the user asked to process EDF files for chb01-chb04.
            # Processing all of them might take too long. Let's limit to files WITH seizures and 1 without.
            has_seizure = len(seizures_dict.get(edf_filename, [])) > 0
            if not has_seizure and edf_filename != patient_records[0].split('/')[1]:
                # Skip to save time, only process the first file (usually background) and files with seizures
                continue
                
            edf_url = f"{BASE_URL}/{record}"
            edf_path = patient_dir / edf_filename
            download_file(edf_url, edf_path)
            
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                data = raw.get_data() # shape (n_channels, n_samples)
                ch_names = raw.ch_names
                
                n_samples = data.shape[1]
                n_windows = n_samples // SAMPLES_PER_WINDOW
                
                sz_times = seizures_dict.get(edf_filename, [])
                
                for w in range(n_windows):
                    start_idx = w * SAMPLES_PER_WINDOW
                    end_idx = start_idx + SAMPLES_PER_WINDOW
                    
                    window_start_sec = start_idx / FS
                    window_end_sec = end_idx / FS
                    
                    # Target is 1 if window overlaps with any seizure
                    target = 0
                    for (sz_start, sz_end) in sz_times:
                        if (window_start_sec <= sz_end) and (window_end_sec >= sz_start):
                            target = 1
                            break
                            
                    window_data = data[:, start_idx:end_idx]
                    
                    # Use the first channel for now to match our generic single-channel logic, or extract all?
                    # "matching our existing feature approach". The Bonn approach uses 1 channel.
                    # CHB-MIT has 23 channels. Let's just use channel 0 to match Bonn single channel, 
                    # or mean over channels, or extract all. The CHB-MIT notebook originally used all channels?
                    # Let's just extract features for channel 0 for this quick script.
                    features = extract_features_multichannel(
                        window_data[0:1, :], 
                        channel_names=["Ch0"], 
                        fs=FS
                    )
                    
                    features['target'] = target
                    features['patient_id'] = patient
                    features['record'] = edf_filename
                    features['window_idx'] = w
                    
                    rows.append(features)
            except Exception as e:
                print(f"Failed to process {edf_filename}: {e}")
            finally:
                # Delete EDF to save space
                if edf_path.exists():
                    edf_path.unlink()

    if rows:
        df = pd.DataFrame(rows)
        out_parquet = Path("chbmit_subset.parquet")
        df.to_parquet(out_parquet, index=False)
        print(f"Successfully saved {len(df)} windows to {out_parquet}")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        print(f"Patient distribution: {df['patient_id'].value_counts().to_dict()}")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()
