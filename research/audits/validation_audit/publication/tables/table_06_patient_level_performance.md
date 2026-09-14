### Table 6: Patient-Specific Seizure Detection and False Alarm Breakdown

| patient_id | model_id | architecture_name | total_windows | recording_hours | num_seizures | detected_seizures | missed_seizures | event_sensitivity | window_sensitivity | window_specificity | window_precision | f1_score | auroc | auprc | false_alarms_count | false_alarms_per_day | mean_detection_delay_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chb01 | Model A | 1D CNN Baseline | 58353 | 40.55 | 7 | 3 | 4 | 0.4286 | 0.11111 | 0.99964 | 0.4878 | 0.181 | 0.44356 | 0.10239 | 21 | 12.43 | 17.17 |
| chb01 | Model B | CNN + Spatial GNN (theta=0.30) | 58353 | 40.55 | 7 | 0 | 7 | 0.0 | 0.0 | 0.99988 | 0.0 | 0.0 | 0.06795 | 0.00165 | 7 | 4.14 | nan |
| chb01 | Model C | CNN + Spatial GNN + Causal GRU (L=8) | 58353 | 40.55 | 7 | 6 | 1 | 0.8571 | 0.79444 | 0.99985 | 0.94079 | 0.86145 | 0.98476 | 0.91955 | 9 | 5.33 | 9.25 |
| chb02 | Model A | 1D CNN Baseline | 50747 | 35.27 | 3 | 2 | 1 | 0.6667 | 0.32857 | 0.99169 | 0.0518 | 0.08949 | 0.71703 | 0.16161 | 421 | 286.51 | 19.0 |
| chb02 | Model B | CNN + Spatial GNN (theta=0.30) | 50747 | 35.27 | 3 | 2 | 1 | 0.6667 | 0.27143 | 0.9999 | 0.79167 | 0.40426 | 0.80459 | 0.32273 | 5 | 3.4 | 9.0 |
| chb02 | Model C | CNN + Spatial GNN + Causal GRU (L=8) | 50747 | 35.27 | 3 | 3 | 0 | 1.0 | 0.88571 | 0.99931 | 0.63918 | 0.74251 | 0.98259 | 0.75586 | 35 | 23.82 | 9.67 |
| chb03 | Model A | 1D CNN Baseline | 54684 | 38.0 | 7 | 2 | 5 | 0.2857 | 0.05521 | 0.99963 | 0.31034 | 0.09375 | 0.26441 | 0.03718 | 20 | 12.63 | 8.5 |
| chb03 | Model B | CNN + Spatial GNN (theta=0.30) | 54684 | 38.0 | 7 | 0 | 7 | 0.0 | 0.0 | 0.99895 | 0.0 | 0.0 | 0.09193 | 0.00158 | 57 | 36.0 | nan |
| chb03 | Model C | CNN + Spatial GNN + Causal GRU (L=8) | 54684 | 38.0 | 7 | 7 | 0 | 1.0 | 0.81595 | 0.99824 | 0.58079 | 0.67857 | 0.99774 | 0.70358 | 96 | 60.63 | 9.86 |
| chb05 | Model A | 1D CNN Baseline | 56125 | 39.0 | 5 | 5 | 0 | 1.0 | 0.22321 | 0.78653 | 0.00417 | 0.00819 | 0.27279 | 0.06988 | 11933 | 7342.86 | 1.7 |
| chb05 | Model B | CNN + Spatial GNN (theta=0.30) | 56125 | 39.0 | 5 | 4 | 1 | 0.8 | 0.03571 | 0.97841 | 0.00658 | 0.01112 | 0.17176 | 0.00585 | 1207 | 742.72 | 6.12 |
| chb05 | Model C | CNN + Spatial GNN + Causal GRU (L=8) | 56125 | 39.0 | 5 | 5 | 0 | 1.0 | 0.875 | 0.99537 | 0.43077 | 0.57732 | 0.99281 | 0.86402 | 259 | 159.37 | 13.7 |
