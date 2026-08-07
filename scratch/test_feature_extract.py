import numpy as np
import mne
import sys

# Load CHBMIT model features
import json
with open("apps/api/models/chbmit/selected_features.json", "r") as f:
    selected = json.load(f)

print("Selected features start with:", selected[:3])

# Load predict.py logic
raw = mne.io.read_raw_edf("data/chbmit_subset/chb01/chb01_01.edf", preload=True, verbose=False)
df = raw.to_data_frame()
df.drop(columns=["time"], inplace=True, errors="ignore")
eeg_data = df.values.T

print("eeg_data shape:", eeg_data.shape)

from app.services.pipelines.chbmit.feature_extraction import extract_all_features

data = eeg_data[0:1, 0:15360]
print("data shape before extract:", data.shape)

feature_dict = extract_all_features(data, ["Ch0"], 256.0)
print("Keys in feature_dict:", list(feature_dict.keys())[:5])

# Find missing
missing = set(selected) - set(feature_dict.keys())
print("Missing:", missing)

vector = []
for feat in selected:
    vector.append(feature_dict.get(feat, 0.0))
print("Zeros count:", vector.count(0.0))

