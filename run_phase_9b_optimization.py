import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from itertools import product
from datetime import datetime

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from research.experiments.imbalance.patient_splitter import PatientDataSplitter

# 1. Inputs
NPZ_PATH = os.path.join(BASE_DIR, "research/experiments/model_c/experiments/L8/val_predictions.npz")
CSV_PATH = os.path.join(BASE_DIR, "research/experiments/temporal_post_processing/validation_predictions.csv")
EVENTS_PATH = os.path.join(BASE_DIR, "data/manifests/chbmit_seizure_events.csv")
MANIFEST_PATH = os.path.join(BASE_DIR, "data/manifests/chbmit_window_index.csv.gz")

OUT_DIR = os.path.join(BASE_DIR, "research/experiments/temporal_post_processing")

# Validation parameters
VAL_PATIENTS = ["chb06", "chb07", "chb08", "chb10"]

# 2. Load Data
npz_data = np.load(NPZ_PATH)
y_prob_raw = npz_data["y_prob"]

df = pd.read_csv(CSV_PATH)

# Verify order and mapping
assert len(df) == len(y_prob_raw)
assert sorted(df["patient_id"].unique().tolist()) == VAL_PATIENTS
assert "chb01" not in df["patient_id"].unique()

# Prefer NPZ float32 over CSV 6-decimal rounded values
df["pred_prob"] = y_prob_raw

events_df = pd.read_csv(EVENTS_PATH)
events_df = events_df[events_df["patient_id"].isin(VAL_PATIENTS)].copy()
total_events_overall = len(events_df)

# Group dataframe by recording
recordings = {rec_id: group for rec_id, group in df.groupby("recording_id")}

total_duration_hours = len(df) * 2.5 / 3600.0

# 3. Post-processing logic
def process_recording(rec_df: pd.DataFrame, n: int, m: int, min_dur: float, merge_int: float, refract: float) -> List[Tuple[float, float]]:
    raw = (rec_df["pred_prob"].values >= 0.50).astype(int)
    if m == 1:
        triggers = raw
    else:
        triggers = np.zeros(len(raw), dtype=int)
        for i in range(len(raw)):
            start = max(0, i - m + 1)
            if np.sum(raw[start:i+1]) >= n:
                triggers[i] = 1
                
    diffs = np.diff(np.concatenate(([0], triggers, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    
    if len(starts) == 0:
        return []
        
    starts_sec = rec_df["window_start_sec"].values
    ends_sec = rec_df["window_end_sec"].values
    
    segs = [[starts_sec[s], ends_sec[e]] for s, e in zip(starts, ends)]
    
    merged = []
    for seg in segs:
        if not merged:
            merged.append(seg)
        else:
            last_seg = merged[-1]
            if seg[0] - last_seg[1] <= merge_int + 1e-5:
                last_seg[1] = seg[1]
            else:
                merged.append(seg)
                
    dur_filtered = []
    for seg in merged:
        if seg[1] - seg[0] >= min_dur - 1e-5:
            dur_filtered.append(seg)
            
    final_segs = []
    last_end = -float('inf')
    for seg in dur_filtered:
        if seg[0] >= last_end + refract - 1e-5:
            final_segs.append(seg)
            last_end = seg[1]
            
    return final_segs

def evaluate_config(n, m, min_dur, merge_int, refract):
    total_episodes = 0
    tp_alarms = 0
    fp_alarms = 0
    detected_events = 0
    
    delays = []
    
    # Pre-filter events per recording for speed
    # Compute alarms
    for rec_id, rec_df in recordings.items():
        alarms = process_recording(rec_df, n, m, min_dur, merge_int, refract)
        total_episodes += len(alarms)
        
        sz_evts = events_df[events_df["recording_id"] == rec_id]
        
        # Match alarms to events
        rec_fp = 0
        rec_tp = 0
        
        # Which events got detected?
        ev_detected = set()
        
        for al_s, al_e in alarms:
            matched = False
            for _, ev in sz_evts.iterrows():
                sz_s = ev["start_sec"]
                sz_e = ev["end_sec"]
                
                # Overlap condition
                if al_e > sz_s and al_s < sz_e:
                    matched = True
                    if ev["seizure_id"] not in ev_detected:
                        ev_detected.add(ev["seizure_id"])
                        delay = max(0.0, float(al_e - sz_s)) # first overlapping alarm end - sz_start
                        delays.append(delay)
            if matched:
                rec_tp += 1
            else:
                rec_fp += 1
                
        tp_alarms += rec_tp
        fp_alarms += rec_fp
        detected_events += len(ev_detected)
        
    sens = detected_events / total_events_overall if total_events_overall > 0 else 0
    fa_per_day = fp_alarms / (total_duration_hours / 24.0)
    
    median_delay = np.median(delays) if delays else float('nan')
    
    return {
        "n": n, "m": m, "min_dur": min_dur, "merge_int": merge_int, "refract": refract,
        "total_episodes": total_episodes,
        "tp_alarms": tp_alarms,
        "fp_alarms": fp_alarms,
        "fa_per_day": fa_per_day,
        "detected_events": detected_events,
        "total_events": total_events_overall,
        "event_sens": sens,
        "median_delay": median_delay
    }

# 4. Search Space
# Stage A Candidates: Persistence/N-of-M
n_of_m_cands = [(1,1), (2,2), (3,3), (4,4), (5,5), (2,3), (3,4), (3,5), (4,5), (4,6), (5,7)]
merge_cands = [0, 5, 10, 15, 20, 30, 60]
dur_cands = [0, 5, 7.5, 10, 15]
refract_cands = [0, 15, 30, 60, 120]

print(f"Total Validation Recordings: {len(recordings)}")
print(f"Total Seizure Events: {total_events_overall}")
print("Evaluating Baseline (1 of 1, 0 merge, 0 dur, 0 refract)...")
baseline = evaluate_config(1, 1, 0, 0, 0)
print(f"Baseline -> Sens: {baseline['event_sens']*100:.1f}%, FA/day: {baseline['fa_per_day']:.2f}")

print("Running Stage A (N-of-M only)...")
results = []
for n, m in n_of_m_cands:
    res = evaluate_config(n, m, 0, 0, 0)
    results.append(res)
    
df_a = pd.DataFrame(results)
# Pick top 5 N-of-M that keep Sens >= 0.90
valid_a = df_a[df_a["event_sens"] >= 0.90].copy()
valid_a.sort_values(by="fa_per_day", inplace=True)
best_n_m = valid_a.head(3)[["n", "m"]].values.tolist()

if not best_n_m:
    print("WARNING: No N-of-M candidates preserved 90% sens. Expanding search.")
    best_n_m = [(1,1), (2,3), (3,4)]

print("Running Stage B (Combinations)...")
all_cands = []
for (n, m), min_d, merge, ref in product(best_n_m, dur_cands, merge_cands, refract_cands):
    # Only sensible combinations
    all_cands.append((n, m, min_d, merge, ref))

print(f"Total Stage B combinations: {len(all_cands)}")
stage_b_results = []
for i, c in enumerate(all_cands):
    if i % 100 == 0:
        print(f"  {i}/{len(all_cands)}")
    res = evaluate_config(*c)
    stage_b_results.append(res)

all_results = pd.DataFrame(stage_b_results)
all_results.to_csv(os.path.join(OUT_DIR, "phase_9b_candidates_raw.csv"), index=False)

# Selection Rule
# Min FA/day where Sens >= 90%
valid = all_results[all_results["event_sens"] >= 0.90].copy()
valid.sort_values(by=["fa_per_day", "event_sens", "median_delay"], ascending=[True, False, True], inplace=True)

if len(valid) == 0:
    print("CRITICAL WARNING: No config maintained 90% sensitivity!")
    best = all_results.sort_values(by=["event_sens", "fa_per_day"], ascending=[False, True]).iloc[0]
else:
    best = valid.iloc[0]

print("SELECTED CONFIGURATION:")
print(best)

# Save Selected Config
sel = {
    "probability_threshold": 0.50,
    "n_of_m": f"{int(best['n'])}-of-{int(best['m'])}",
    "min_duration_sec": float(best["min_dur"]),
    "merge_interval_sec": float(best["merge_int"]),
    "refractory_period_sec": float(best["refract"]),
    "event_matching_tolerance": "strict window overlap",
    "selection_criterion": "Minimize FA/day s.t. Event Sensitivity >= 0.90",
    "validation_metrics": {
        "event_sensitivity": float(best["event_sens"]),
        "false_alarms_per_day": float(best["fa_per_day"]),
        "median_delay_sec": float(best["median_delay"]),
        "total_alarms": int(best["total_episodes"]),
        "true_positive_alarms": int(best["tp_alarms"]),
        "false_positive_alarms": int(best["fp_alarms"])
    },
    "timestamp": datetime.utcnow().isoformat() + "Z"
}
with open(os.path.join(OUT_DIR, "phase_9b_selected_postprocessing.json"), "w") as f:
    json.dump(sel, f, indent=4)

