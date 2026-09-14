import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

out_dir = "research/audit/model_c_validation"

# Data
test_df = pd.read_csv("research/experiments/model_c/results/final_test_predictions.csv")
val_df = pd.read_csv("research/experiments/temporal_post_processing/validation_predictions.csv")

test_ev = pd.read_csv(f"{out_dir}/test_event_summary.csv")
val_ev = pd.read_csv(f"{out_dir}/validation_event_summary.csv")

test_df["dataset"] = "Test"
val_df["dataset"] = "Validation"

# Ensure label columns are present
if "label_50pct_overlap" in test_df.columns:
    test_df.rename(columns={"label_50pct_overlap": "label"}, inplace=True)
if "label_50pct_overlap" in val_df.columns:
    val_df.rename(columns={"label_50pct_overlap": "label"}, inplace=True)

# 1. Validation seizure/non-seizure prob dist
plt.figure(figsize=(10, 6))
sns.histplot(data=val_df, x="predicted_probability", hue="label", bins=50, log_scale=(False, True))
plt.title("Validation Seizure vs Non-Seizure Probability Distribution")
plt.savefig(f"{out_dir}/val_prob_dist.png")
plt.close()

# 2. Test seizure/non-seizure prob dist
plt.figure(figsize=(10, 6))
sns.histplot(data=test_df, x="predicted_probability", hue="label", bins=50, log_scale=(False, True))
plt.title("Test Seizure vs Non-Seizure Probability Distribution")
plt.savefig(f"{out_dir}/test_prob_dist.png")
plt.close()

# 3. Per-patient validation event sensitivity
plt.figure(figsize=(10, 6))
val_sens = val_ev.groupby("patient")["detected"].mean().reset_index()
sns.barplot(data=val_sens, x="patient", y="detected")
plt.title("Per-patient Validation Event Sensitivity")
plt.ylabel("Event Sensitivity")
plt.savefig(f"{out_dir}/val_patient_sensitivity.png")
plt.close()

# 4. Per-patient test event sensitivity
plt.figure(figsize=(10, 6))
test_sens = test_ev.groupby("patient")["detected"].mean().reset_index()
sns.barplot(data=test_sens, x="patient", y="detected")
plt.title("Per-patient Test Event Sensitivity")
plt.ylabel("Event Sensitivity")
plt.savefig(f"{out_dir}/test_patient_sensitivity.png")
plt.close()

# 5. Per-event detection delay
all_ev = pd.concat([test_ev, val_ev])
detected_ev = all_ev[all_ev["detected"] == True]
plt.figure(figsize=(12, 6))
sns.barplot(data=detected_ev, x="recording", y="delay")
plt.title("Per-Event Detection Delay (Seconds)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{out_dir}/per_event_delay.png")
plt.close()

# 6. Validation vs test probability distribution
combined_df = pd.concat([test_df, val_df])
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, x="predicted_probability", hue="dataset", common_norm=False)
plt.title("Validation vs Test Probability Distribution")
plt.savefig(f"{out_dir}/val_vs_test_prob_dist.png")
plt.close()

# 7. False-positive probability distribution
fp_df = combined_df[(combined_df["label"] == 0) & (combined_df["predicted_probability"] >= 0.5)]
plt.figure(figsize=(10, 6))
sns.histplot(data=fp_df, x="predicted_probability", bins=30)
plt.title("False Positive Probability Distribution (Prob >= 0.5)")
plt.savefig(f"{out_dir}/false_positive_prob_dist.png")
plt.close()

print("Figures generated successfully.")
