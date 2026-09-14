import os
import json
import numpy as np
import scipy.signal as signal
import mne
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import hashlib

# Need to ensure proper path
import sys
sys.path.append("/home/bhumi/GitHub/NeuroAegis")
from research.experiments.windowing_labeling.chbmit_preprocessor import CHBMITChannelManager, CHBMITSignalFilter, CHBMITNormalizer

out_dir = Path("research/visualizations/eeg_preprocessing_example")
out_dir.mkdir(parents=True, exist_ok=True)

edf_path = "data/CHB-MIT Dataset/chb01/chb01_03.edf"
sfreq = 256.0

# Seizure is 2996 to 3036. 10 seconds around start.
start_sec = 2993
end_sec = start_sec + 10
start_sample = int(start_sec * sfreq)
end_sample = int(end_sec * sfreq)
duration_samples = end_sample - start_sample

cm = CHBMITChannelManager()
raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
picked_indices, montage_status, missing = cm.map_recording_channels(raw.ch_names)

raw_data, _ = raw[picked_indices, start_sample:end_sample] # Shape: (23, 2560)
raw_data_uv = raw_data * 1e6

target_channel_name = "FP1-F7"
target_channel_idx = cm.get_canonical_channels().index(target_channel_name)

raw_channel_data_uv = raw_data_uv[target_channel_idx]
time_axis = np.linspace(start_sec, end_sec, duration_samples, endpoint=False)

sf = CHBMITSignalFilter(lowcut=0.5, highcut=40.0, notch_freq=60.0, notch_q=30.0, sfreq=256.0, order=4)
bandpass_data_uv = signal.sosfiltfilt(sf.sos_bp, raw_channel_data_uv)
notch_data_uv = signal.filtfilt(sf.b_notch, sf.a_notch, bandpass_data_uv)

norm = CHBMITNormalizer()
# We will do windowing exactly as model input expects. The model takes 5-sec windows.
# The `chbmit_preprocessor` does zscore_recording_local on the output window.
# Let's extract the 5-sec window: 2996 to 3001
win_start_sec = 2996
win_end_sec = 3001
win_start_samp = int((win_start_sec - start_sec) * sfreq)
win_end_samp = int((win_end_sec - start_sec) * sfreq)

win_raw = raw_data_uv[:, win_start_samp:win_end_samp]
# filter the whole 10s for continuity, then slice
bandpass_all_uv = signal.sosfiltfilt(sf.sos_bp, raw_data_uv, axis=-1)
notch_all_uv = signal.filtfilt(sf.b_notch, sf.a_notch, bandpass_all_uv, axis=-1)
win_notch = notch_all_uv[:, win_start_samp:win_end_samp]
win_norm = norm.zscore_recording_local(win_notch)

norm_mean = np.mean(win_notch[target_channel_idx])
norm_std = np.std(win_notch[target_channel_idx])

# Now calculate PSD for 10s segment
def compute_psd(data, fs):
    f, pxx = signal.welch(data, fs, nperseg=1024)
    return f, pxx

f_raw, pxx_raw = compute_psd(raw_channel_data_uv, sfreq)
f_bp, pxx_bp = compute_psd(bandpass_data_uv, sfreq)
f_notch, pxx_notch = compute_psd(notch_data_uv, sfreq)

idx_60 = (f_raw >= 58) & (f_raw <= 62)
p60_raw = np.sum(pxx_raw[idx_60])
p60_bp = np.sum(pxx_bp[idx_60])
p60_notch = np.sum(pxx_notch[idx_60])
reduction_bp = (p60_raw - p60_bp) / p60_raw * 100 if p60_raw > 0 else 0
reduction_notch = (p60_bp - p60_notch) / p60_bp * 100 if p60_bp > 0 else 0

# Figures
# Fig 1: Complete pipeline (Raw -> BP -> Notch -> Norm) for the 5-sec window
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
win_time = np.linspace(win_start_sec, win_end_sec, win_end_samp - win_start_samp, endpoint=False)
axes[0].plot(win_time, win_raw[target_channel_idx], color='black')
axes[0].set_title(f"1. Raw EEG Signal ({target_channel_name}) [µV]")
axes[1].plot(win_time, bandpass_all_uv[target_channel_idx, win_start_samp:win_end_samp], color='blue')
axes[1].set_title("2. Bandpass Filtered (0.5-40 Hz) [µV]")
axes[2].plot(win_time, win_notch[target_channel_idx], color='green')
axes[2].set_title("3. Notch Filtered (60 Hz) [µV]")
axes[3].plot(win_time, win_norm[target_channel_idx], color='red')
axes[3].set_title(f"4. Normalized (per-channel z-score, $\mu={norm_mean:.2f}, \sigma={norm_std:.2f}$)")
axes[3].set_xlabel("Time (seconds)")
plt.tight_layout()
fig.savefig(out_dir / "figure_1_complete_pipeline.png")
plt.close(fig)

# Fig 2: Time-Domain Before/After Comparison
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(time_axis, raw_channel_data_uv, color='black', label="Raw")
axes[0].legend()
axes[0].set_title("Time-Domain Comparison (10s)")
axes[1].plot(time_axis, bandpass_data_uv, color='blue', label="Bandpass (0.5-40Hz)")
axes[1].legend()
axes[2].plot(time_axis, notch_data_uv, color='green', label="Notch (60Hz)")
axes[2].legend()
axes[2].set_xlabel("Time (seconds)")
plt.tight_layout()
fig.savefig(out_dir / "figure_2_time_domain.png")
plt.close(fig)

# Fig 3: Frequency-Domain
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].semilogy(f_raw, pxx_raw, color='black')
axes[0].set_title("Raw PSD")
axes[1].semilogy(f_bp, pxx_bp, color='blue')
axes[1].set_title("Bandpass PSD")
axes[2].semilogy(f_notch, pxx_notch, color='green')
axes[2].set_title("Notch PSD")
axes[2].set_xlabel("Frequency (Hz)")
plt.tight_layout()
fig.savefig(out_dir / "figure_3_frequency_domain.png")
plt.close(fig)

# Fig 4: Notch Analysis
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(f_bp, pxx_bp, color='blue', label='Bandpass only')
ax.semilogy(f_notch, pxx_notch, color='green', label='Bandpass + Notch')
ax.set_xlim(50, 70)
ax.set_title("60 Hz Notch Analysis (50-70 Hz Zoom)")
ax.set_xlabel("Frequency (Hz)")
ax.legend()
plt.tight_layout()
fig.savefig(out_dir / "figure_4_notch_analysis.png")
plt.close(fig)

# Fig 5: Model Input Heatmap
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(win_norm, aspect='auto', interpolation='none', cmap='viridis')
ax.set_yticks(np.arange(23))
ax.set_yticklabels(cm.get_canonical_channels())
ax.set_xlabel("Samples (0-1280)")
ax.set_title("Actual Model Input — 23 Channels × 1,280 Samples")
plt.colorbar(im, ax=ax)
plt.tight_layout()
fig.savefig(out_dir / "figure_5_model_input.png")
plt.close(fig)

# Filter response
b_bp, a_bp = signal.butter(4, [0.5, 40.0], btype="bandpass", fs=256.0)
w_bp, h_bp = signal.freqz(b_bp, a_bp, fs=256.0)
w_n, h_n = signal.freqz(sf.b_notch, sf.a_notch, fs=256.0)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(w_bp, 20 * np.log10(abs(h_bp) + 1e-12), label="Bandpass (0.5-40Hz) IIR Equivalent")
ax.plot(w_n, 20 * np.log10(abs(h_n) + 1e-12), label="Notch (60Hz)")
ax.set_title("Filter Frequency Response")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude (dB)")
ax.set_ylim(-60, 5)
ax.legend()
plt.tight_layout()
fig.savefig(out_dir / "figure_6_filter_response.png")
plt.close(fig)

# Save data
pd.DataFrame({"time": time_axis, "raw": raw_channel_data_uv}).to_csv(out_dir / "raw_signal.csv", index=False)
pd.DataFrame({"time": time_axis, "bandpass": bandpass_data_uv}).to_csv(out_dir / "bandpass_signal.csv", index=False)
pd.DataFrame({"time": time_axis, "notch": notch_data_uv}).to_csv(out_dir / "notch_signal.csv", index=False)
win_time_df = np.linspace(win_start_sec, win_end_sec, win_end_samp - win_start_samp, endpoint=False)
pd.DataFrame({"time": win_time_df, "normalized": win_norm[target_channel_idx]}).to_csv(out_dir / "normalized_signal.csv", index=False)
pd.DataFrame({"time": win_time_df, "window_signal": win_notch[target_channel_idx]}).to_csv(out_dir / "window_signal.csv", index=False)
np.save(out_dir / "model_input.npy", win_norm)

# Write output stats
with open(out_dir / "stats.txt", "w") as f:
    f.write(f"p60_raw: {p60_raw}\n")
    f.write(f"p60_bp: {p60_bp}\n")
    f.write(f"p60_notch: {p60_notch}\n")
    f.write(f"reduction_bp: {reduction_bp}\n")
    f.write(f"reduction_notch: {reduction_notch}\n")

print("Done generating.")
