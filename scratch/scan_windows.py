import mne
import numpy as np
import sys
from app.services.prediction.chbmit_predictor import CHBMITPredictor

predictor = CHBMITPredictor(model_dir="apps/api/models/chbmit", default_model="lightgbm")
predictor.load_model()

raw = mne.io.read_raw_edf("data/chbmit_subset/chb06/chb06_01.edf", preload=True, verbose=False)
df = raw.to_data_frame(scalings=dict(eeg=1))
df.drop(columns=["time"], inplace=True, errors="ignore")
eeg_data = df.values.T

seizure_sec = 1724
start_w = (seizure_sec // 60) - 2

for i in range(5):
    w_sec = (start_w + i) * 60
    start_sample = w_sec * 256
    w_data = eeg_data[:, start_sample:start_sample + 15360]
    
    vec, _ = predictor.extract_features(w_data, ["Ch0"], 256.0)
    pred = predictor.predict(vec)
    print(f"Window {w_sec}s - {w_sec+60}s: Prob = {pred['probabilities']['seizure']:.4f} (Seizure at 1724s: {w_sec <= 1724 <= w_sec+60})")

