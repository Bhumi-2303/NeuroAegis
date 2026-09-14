import pandas as pd
import numpy as np

events = pd.read_csv("data/manifests/chbmit_seizure_events.csv")
def write_mismatches(pred_csv, out_csv):
    df = pd.read_csv(pred_csv)
    if 'label_any_overlap' in df.columns:
        df.rename(columns={"label_any_overlap": "label"}, inplace=True)
        
    df["y_true_recon"] = 0
    df["mid_idx"] = np.arange(len(df))
    sub_events = events[events["recording_id"].isin(df["recording_id"].unique())]
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
        
    mismatches = df[df["y_true_recon"] != df["label"]].copy()
    mismatches = mismatches[["patient_id", "recording_id", "window_start_sec", "window_end_sec", "label", "y_true_recon"]]
    mismatches.to_csv(out_csv, index=False)

write_mismatches("research/experiments/model_c/results/final_test_predictions.csv", "research/audit/model_c_validation/test_mismatches.csv")
write_mismatches("research/experiments/temporal_post_processing/validation_predictions.csv", "research/audit/model_c_validation/val_mismatches.csv")
