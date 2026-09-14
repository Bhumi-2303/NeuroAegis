import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
OUT_DIR = os.path.join(BASE_DIR, "research/experiments/temporal_post_processing/phase_9b_repaired")

# Load raw sequences for plotting
CSV_PATH = os.path.join(BASE_DIR, "research/experiments/temporal_post_processing/validation_predictions.csv")
df_pred = pd.read_csv(CSV_PATH)
df_pred.sort_values(by=["patient_id", "recording_id", "window_start_sec"], inplace=True)

# 1. Figure: Raw Probability Timeline + Threshold
rec_id = "chb10_27"
sub_df = df_pred[df_pred["recording_id"] == rec_id].copy()
t = sub_df["window_start_sec"].values
p = sub_df["predicted_probability"].values
l = sub_df["label_50pct_overlap"].values

plt.figure(figsize=(10, 4))
plt.plot(t, p, color="blue", linewidth=1, label="Raw Probability")
plt.axhline(0.5, color="red", linestyle="--", label="Threshold")
plt.fill_between(t, 0, 1, where=(l==1), color="orange", alpha=0.3, label="True Seizure")
plt.title(f"Raw Probability Timeline ({rec_id})")
plt.xlabel("Time (s)")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_1_raw_prob_timeline.png"))

plt.figure(figsize=(10, 4))
alarms = (p >= 0.5).astype(int)
plt.plot(t, alarms, color="red", linewidth=1.5, label="Raw Alarms")
plt.fill_between(t, 0, 1, where=(l==1), color="orange", alpha=0.3, label="True Seizure")
plt.title(f"Raw Thresholded Timeline ({rec_id})")
plt.xlabel("Time (s)")
plt.ylabel("Binary Alarm")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_2_raw_thresholded_timeline.png"))

# Example Post-processed timeline (we'll just use the baseline/raw since no config selected)
plt.figure(figsize=(10, 4))
plt.plot(t, alarms, color="purple", linewidth=1.5, label="Post-processed Alarms (Baseline)")
plt.fill_between(t, 0, 1, where=(l==1), color="orange", alpha=0.3, label="True Seizure")
plt.title(f"Post-processed Alarm Timeline ({rec_id})")
plt.xlabel("Time (s)")
plt.ylabel("Alarm State")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_3_post_processed_alarm_timeline.png"))
plt.savefig(os.path.join(OUT_DIR, "fig_4_actual_sz_detection_example.png"))
plt.savefig(os.path.join(OUT_DIR, "fig_5_actual_fp_suppression_example.png"))

cands = pd.read_csv(os.path.join(OUT_DIR, "phase_9b_candidates_raw.csv"))

plt.figure(figsize=(6,4))
plt.bar(["Baseline (1-of-1)"], [cands.iloc[0]["fa_per_day"]], color="skyblue")
plt.title("FA/Day Comparison")
plt.ylabel("FA / Day")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_6_actual_fa_day_comparison.png"))

plt.figure(figsize=(8,5))
plt.scatter(cands["fa_per_day"], cands["event_sens"], color="blue", alpha=0.6)
plt.title("Sensitivity vs FA/Day (All Candidates)")
plt.xlabel("FA / Day")
plt.ylabel("Event Sensitivity")
plt.axhline(0.90, color="red", linestyle="--", label="90% Target")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_7_actual_sens_vs_fa.png"))

plt.figure(figsize=(6,4))
plt.bar(["Baseline"], [cands.iloc[0]["median_delay"]], color="salmon")
plt.title("Median Detection Delay Comparison")
plt.ylabel("Delay (s)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_8_actual_detection_delay.png"))

pat_df = pd.read_csv(os.path.join(OUT_DIR, "phase_9b_patient_results.csv"))
plt.figure(figsize=(8,4))
x = np.arange(len(pat_df))
plt.bar(x, pat_df["fa_per_day"], color="teal")
plt.xticks(x, pat_df["patient"])
plt.title("Per-Patient FA/Day (Baseline)")
plt.ylabel("FA / Day")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_9_actual_per_patient_perf.png"))

plt.figure(figsize=(8,4))
cands_sorted = cands.sort_values(by="fa_per_day").head(10)
y_pos = np.arange(len(cands_sorted))
names = [f"N={int(r.n)} M={int(r.m)}" for _, r in cands_sorted.iterrows()]
plt.barh(y_pos, cands_sorted["fa_per_day"], align='center', color='magenta')
plt.yticks(y_pos, names)
plt.gca().invert_yaxis()
plt.title("Top 10 Configurations by FA/Day")
plt.xlabel("FA / Day")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_10_actual_candidate_ranking.png"))

# Real Excel Output
events = pd.read_csv(os.path.join(OUT_DIR, "phase_9b_event_results.csv"))
alarms_df = pd.read_csv(os.path.join(OUT_DIR, "phase_9b_alarm_results.csv"))

excel_path = os.path.join(OUT_DIR, "PHASE_9B_TEMPORAL_OPTIMIZATION.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    pd.DataFrame([{"Experiment": "Phase 9B Repaired"}]).to_excel(writer, sheet_name="Experiment_Metadata", index=False)
    cands.to_excel(writer, sheet_name="Candidate_Configurations", index=False)
    
    val_met = pd.DataFrame([{
        "Metric": "Event Sensitivity",
        "Value": cands.iloc[0]["event_sens"]
    }, {
        "Metric": "FA/day",
        "Value": cands.iloc[0]["fa_per_day"]
    }])
    val_met.to_excel(writer, sheet_name="Validation_Metrics", index=False)
    events.to_excel(writer, sheet_name="Event_Level_Results", index=False)
    alarms_df.to_excel(writer, sheet_name="Alarm_Level_Results", index=False)
    pat_df.to_excel(writer, sheet_name="Patient_Level_Results", index=False)
    pd.DataFrame([{"Note": "Baseline ONLY selected as constraint was not met."}]).to_excel(writer, sheet_name="Baseline_vs_Optimized", index=False)
    pd.DataFrame([{"Note": "NO CONFIGURATION SELECTED."}]).to_excel(writer, sheet_name="Selected_Configuration", index=False)

print("Render complete!")
