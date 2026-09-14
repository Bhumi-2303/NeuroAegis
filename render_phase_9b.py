import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
OUT_DIR = os.path.join(BASE_DIR, "research/experiments/temporal_post_processing")

# 1. Load data
candidates_df = pd.read_csv(os.path.join(OUT_DIR, "phase_9b_candidates_raw.csv"))
with open(os.path.join(OUT_DIR, "phase_9b_selected_postprocessing.json"), "r") as f:
    selected = json.load(f)

# Sort candidates by Sens DESC, FA ASC
candidates_df.sort_values(by=["event_sens", "fa_per_day"], ascending=[False, True], inplace=True)

best = candidates_df.iloc[0]
baseline = candidates_df[(candidates_df["n"] == 1) & (candidates_df["m"] == 1) & 
                         (candidates_df["min_dur"] == 0) & (candidates_df["merge_int"] == 0) & 
                         (candidates_df["refract"] == 0)].iloc[0]

# 2. Excel Output
excel_path = os.path.join(OUT_DIR, "PHASE_9B_TEMPORAL_OPTIMIZATION.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    # 1. Experiment_Metadata
    pd.DataFrame([{
        "experiment": "Phase 9B Temporal Post-Processing",
        "timestamp": datetime.utcnow().isoformat(),
        "total_combinations": len(candidates_df),
        "validation_patients": "chb06, chb07, chb08, chb10",
        "baseline_fa_per_day": baseline["fa_per_day"],
        "optimized_fa_per_day": best["fa_per_day"],
        "baseline_sens": baseline["event_sens"],
        "optimized_sens": best["event_sens"]
    }]).to_excel(writer, sheet_name="Experiment_Metadata", index=False)
    
    # 2. Candidate_Configurations
    candidates_df.to_excel(writer, sheet_name="Candidate_Configurations", index=False)
    
    # 3. Validation_Metrics
    # Basic metrics mapped
    pd.DataFrame([{
        "Metric": "Event Sensitivity",
        "Baseline": baseline["event_sens"],
        "Optimized": best["event_sens"]
    }, {
        "Metric": "FA/day",
        "Baseline": baseline["fa_per_day"],
        "Optimized": best["fa_per_day"]
    }, {
        "Metric": "Median Delay (s)",
        "Baseline": baseline["median_delay"],
        "Optimized": best["median_delay"]
    }]).to_excel(writer, sheet_name="Validation_Metrics", index=False)
    
    # 4. Event_Level_Results
    pd.DataFrame([{
        "Configuration": "Baseline",
        "Total_Events": baseline["total_events"],
        "Detected": baseline["detected_events"],
        "Missed": baseline["total_events"] - baseline["detected_events"],
        "Sensitivity": baseline["event_sens"]
    }, {
        "Configuration": "Optimized",
        "Total_Events": best["total_events"],
        "Detected": best["detected_events"],
        "Missed": best["total_events"] - best["detected_events"],
        "Sensitivity": best["event_sens"]
    }]).to_excel(writer, sheet_name="Event_Level_Results", index=False)
    
    # 5. Alarm_Level_Results
    pd.DataFrame([{
        "Configuration": "Baseline",
        "Total_Alarms": baseline["total_episodes"],
        "True_Positive_Alarms": baseline["tp_alarms"],
        "False_Positive_Alarms": baseline["fp_alarms"],
        "FA_per_day": baseline["fa_per_day"]
    }, {
        "Configuration": "Optimized",
        "Total_Alarms": best["total_episodes"],
        "True_Positive_Alarms": best["tp_alarms"],
        "False_Positive_Alarms": best["fp_alarms"],
        "FA_per_day": best["fa_per_day"]
    }]).to_excel(writer, sheet_name="Alarm_Level_Results", index=False)
    
    # 6. Patient_Level_Results (Stub for now, needs full processing per patient if required)
    pd.DataFrame({"Note": ["Aggregate evaluated in primary script. See report."]}).to_excel(writer, sheet_name="Patient_Level_Results", index=False)
    
    # 7. Baseline_vs_Optimized
    reduction = 100 * (baseline["fa_per_day"] - best["fa_per_day"]) / baseline["fa_per_day"] if baseline["fa_per_day"] > 0 else 0
    pd.DataFrame([{
        "Metric": "FA/day Reduction (%)",
        "Value": reduction
    }]).to_excel(writer, sheet_name="Baseline_vs_Optimized", index=False)
    
    # 8. Selected_Configuration
    pd.DataFrame([selected]).to_excel(writer, sheet_name="Selected_Configuration", index=False)

# 3. Figures
sns.set_style("whitegrid")

# Fig 6/7: Candidate ranking FA vs Sens
plt.figure(figsize=(10, 6))
sns.scatterplot(data=candidates_df, x="fa_per_day", y="event_sens", alpha=0.6)
plt.scatter([baseline["fa_per_day"]], [baseline["event_sens"]], color='red', s=100, label='Baseline', marker='X')
plt.scatter([best["fa_per_day"]], [best["event_sens"]], color='green', s=100, label='Optimized', marker='*')
plt.title("Candidate Configurations: Sensitivity vs False Alarms/Day")
plt.xlabel("False Alarms per Day")
plt.ylabel("Seizure Event Sensitivity")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_7_sens_vs_fa.png"), dpi=300)

plt.figure(figsize=(10, 6))
top_cands = candidates_df.head(20).copy()
top_cands["name"] = top_cands.apply(lambda r: f"{int(r['n'])}_{int(r['m'])}_{r['min_dur']}_{r['merge_int']}_{r['refract']}", axis=1)
sns.barplot(data=top_cands, x="fa_per_day", y="name", orient="h")
plt.title("Top 20 Configurations by FA/day (Maintaining Best Sensitivity)")
plt.xlabel("FA/day")
plt.ylabel("Configuration (N_M_Dur_Merge_Refract)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_10_candidate_ranking.png"), dpi=300)

print("Render complete!")
