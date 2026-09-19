# NeuroAegis Experiment 6B — Calibration Protocol

## 1. Strict Patient-Level Isolation Architecture
- **Calibration Partition**: Patient `PN00` (Recordings: `PN00-1`, `PN00-2`, `PN00-3`, `PN00-4`, `PN00-5`).
- **Held-Out Test Partition**: Patient `PN12` (Recording: `PN12-3`).
- **Prohibited Operations**: Zero model retraining, zero graph modification, zero held-out threshold tuning.

## 2. Primary Calibration Rule
$$\tau^* = \arg\max_{\tau \in [0.05, 0.95]} F_1(\text{PN00}) \quad \text{subject to} \quad \text{EventSens}(\text{PN00}) \ge 90\%$$

## 3. Tie-Breaking Order
1. Higher Event Sensitivity
2. Higher F1 Score
3. Lower Clinical Alarm Episodes / 24h
4. Lower Mean Detection Delay
5. Higher Precision