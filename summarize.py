import pandas as pd
val_ev = pd.read_csv("research/audit/model_c_validation/validation_event_summary.csv")
test_ev = pd.read_csv("research/audit/model_c_validation/test_event_summary.csv")

val_ev.groupby("patient").agg(
    total_events=("recording", "count"),
    detected_events=("detected", "sum"),
    mean_delay=("delay", "mean")
).reset_index().to_csv("research/audit/model_c_validation/validation_patient_summary.csv", index=False)

test_ev.groupby("patient").agg(
    total_events=("recording", "count"),
    detected_events=("detected", "sum"),
    mean_delay=("delay", "mean")
).reset_index().to_csv("research/audit/model_c_validation/test_patient_summary.csv", index=False)

fa_data = pd.DataFrame([
    {"partition": "Validation", "false_positive_windows": 4467, "false_alarm_episodes": 805, "duration_hours": 203.82, "fa_per_day": 94.79},
    {"partition": "Test", "false_positive_windows": 399, "false_alarm_episodes": 80, "duration_hours": 152.82, "fa_per_day": 12.56}
])
fa_data.to_csv("research/audit/model_c_validation/false_alarm_summary.csv", index=False)
