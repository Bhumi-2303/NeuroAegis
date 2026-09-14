import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import roc_auc_score, average_precision_score

out_dir = "research/audit/model_c_validation"

# Load data
manifest = pd.read_csv("data/manifests/chbmit_manifest.csv")
events = pd.read_csv("data/manifests/chbmit_seizure_events.csv")
events["seizure_id"] = events["patient_id"] + "_" + events.index.astype(str)

def get_metrics(df, events_df, prefix=""):
    # 1. Faster overlap
    df = df.copy()
    df["y_true_recon"] = 0
    df["mid_idx"] = np.arange(len(df))
    
    # Pre-filter events
    sub_events = events_df[events_df["recording_id"].isin(df["recording_id"].unique())]
    
    # Vectorized check
    for rec_id in df["recording_id"].unique():
        rec_evs = sub_events[sub_events["recording_id"] == rec_id]
        if len(rec_evs) == 0: continue
        
        mask = df["recording_id"] == rec_id
        w_start = df.loc[mask, "window_start_sec"].values
        w_end = df.loc[mask, "window_end_sec"].values
        
        y_recon = np.zeros(len(w_start), dtype=int)
        for _, ev in rec_evs.iterrows():
            o_start = np.maximum(w_start, ev["start_sec"])
            o_end = np.minimum(w_end, ev["end_sec"])
            overlap = np.maximum(0, o_end - o_start)
            y_recon[overlap >= 2.5] = 1
        df.loc[mask, "y_true_recon"] = y_recon

    mismatches = df[df["y_true_recon"] != df["label"]]
    mismatch_pct = len(mismatches) / len(df)
    
    y_true = df["y_true_recon"].values
    y_prob = df["predicted_probability"].values
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    sens = tp / (tp + fn) if (tp+fn)>0 else 0
    spec = tn / (tn + fp) if (tn+fp)>0 else 0
    prec = tp / (tp + fp) if (tp+fp)>0 else 0
    f1 = 2 * (prec * sens) / (prec + sens) if (prec+sens)>0 else 0
    bal_acc = (sens + spec) / 2
    
    try: auroc = roc_auc_score(y_true, y_prob)
    except: auroc = np.nan
    try: auprc = average_precision_score(y_true, y_prob)
    except: auprc = np.nan
    
    # Event-level
    delays = []
    det_events = 0
    event_details = []
    
    for _, ev in sub_events.iterrows():
        rec_id = ev["recording_id"]
        sz_start, sz_end = ev["start_sec"], ev["end_sec"]
        
        w_ev = df[(df["recording_id"] == rec_id) & (df["window_end_sec"] > sz_start) & (df["window_start_sec"] < sz_end)]
        w_det = w_ev[w_ev["predicted_probability"] >= 0.5]
        
        is_det = len(w_det) > 0
        if is_det:
            det_events += 1
            delay = max(0.0, float(w_det["window_start_sec"].min() - sz_start))
            delays.append(delay)
        else:
            delay = None
            
        event_details.append({
            "patient": ev["patient_id"],
            "recording": rec_id,
            "event_idx": ev.name,
            "start": sz_start,
            "end": sz_end,
            "detected": is_det,
            "delay": delay,
            "max_prob": float(w_ev["predicted_probability"].max()) if len(w_ev)>0 else 0,
            "onset_prob": float(w_ev["predicted_probability"].iloc[0]) if len(w_ev)>0 else 0,
            "missed_reason": "Probabilities never exceeded threshold during event" if not is_det else ""
        })
        
    ev_sens = det_events / len(sub_events) if len(sub_events)>0 else 0
    
    # FA Episodes Calculation
    fp_windows = df[(y_true == 0) & (y_pred == 1)]
    fa_episodes = 0
    
    for rec_id in fp_windows["recording_id"].unique():
        rec_fps = fp_windows[fp_windows["recording_id"] == rec_id].sort_values("window_start_sec")
        if len(rec_fps) > 0:
            episodes = 1
            last_end = rec_fps.iloc[0]["window_end_sec"]
            for _, r in rec_fps.iloc[1:].iterrows():
                if r["window_start_sec"] > last_end:
                    episodes += 1
                last_end = max(last_end, r["window_end_sec"])
            fa_episodes += episodes
            
    recs_in_partition = manifest[manifest["recording_id"].isin(df["recording_id"].unique())]
    duration_hours = recs_in_partition["recording_duration_sec"].sum() / 3600.0
    fa_per_day = fa_episodes / duration_hours * 24.0 if duration_hours > 0 else 0
    
    res = {
        "threshold": 0.5,
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total_windows": len(df),
        "total_events": len(sub_events),
        "detected_events": det_events,
        "event_sensitivity": ev_sens,
        "mean_delay": np.mean(delays) if delays else None,
        "median_delay": np.median(delays) if delays else None,
        "min_delay": np.min(delays) if delays else None,
        "max_delay": np.max(delays) if delays else None,
        "pct_detected_leq_5s": sum(d <= 5 for d in delays)/len(delays) if delays else 0,
        "pct_detected_leq_10s": sum(d <= 10 for d in delays)/len(delays) if delays else 0,
        "pct_detected_leq_15s": sum(d <= 15 for d in delays)/len(delays) if delays else 0,
        "pct_detected_leq_30s": sum(d <= 30 for d in delays)/len(delays) if delays else 0,
        "duration_hours": duration_hours,
        "false_positive_windows": int(fp),
        "false_alarm_episodes": int(fa_episodes),
        "fa_per_day": fa_per_day,
        "mismatched_labels": len(mismatches),
        "mismatch_pct": mismatch_pct
    }
    
    pd.DataFrame([res]).to_csv(f"{out_dir}/{prefix}metrics.csv", index=False)
    pd.DataFrame(event_details).to_csv(f"{out_dir}/{prefix}event_summary.csv", index=False)
    return res, pd.DataFrame(event_details), df

# Run on TEST
test_df = pd.read_csv("research/experiments/model_c/results/final_test_predictions.csv")
test_df.rename(columns={"predicted_label": "pred", "label_any_overlap": "label"}, inplace=True) 
if 'label' not in test_df.columns: test_df['label'] = 0
test_res, test_ev_df, test_df_mod = get_metrics(test_df, events, prefix="test_")

# Run on VAL
val_csv_path = "research/experiments/temporal_post_processing/validation_predictions.csv"
if os.path.exists(val_csv_path):
    val_df = pd.read_csv(val_csv_path)
    val_df.rename(columns={"predicted_label": "pred", "label_any_overlap": "label"}, inplace=True)
    if 'label' not in val_df.columns: val_df['label'] = 0
    val_res, val_ev_df, val_df_mod = get_metrics(val_df, events, prefix="validation_")

# Missed test seizure forensics
missed_test = test_ev_df[~test_ev_df["detected"]]
missed_test.to_csv(f"{out_dir}/missed_event_forensics.csv", index=False)

print("Metrics saved.")
