# NeuroAegis Experiment 6 — Cross-Domain Error & Missed Seizure Analysis

## 1. Clinical Seizure Detection Breakdown
- **Evaluated Seizures**: 4 clinical seizures
- **Detected Seizures**: **4 / 4 (100.0% Event Sensitivity)**
- **Missed Seizures**: **0 / 4 (0.0% Miss Rate)**

### Detailed Seizure Detections:
1. **`PN00_sz01`** (Recording: `PN00/PN00-1.edf`, Duration: $70.0\text{s}$, Start: $1143.0\text{s}$):
   - **Detected**: YES at $t=1157.5\text{s}$ (Delay: $14.50\text{s}$)
   - **Peak Probability**: $0.9994$
2. **`PN00_sz04`** (Recording: `PN00/PN00-4.edf`, Duration: $74.0\text{s}$, Start: $1006.0\text{s}$):
   - **Detected**: YES at $t=1022.5\text{s}$ (Delay: $16.50\text{s}$)
   - **Peak Probability**: $0.9989$
3. **`PN00_sz05`** (Recording: `PN00/PN00-5.edf`, Duration: $67.0\text{s}$, Start: $904.0\text{s}$):
   - **Detected**: YES at $t=922.5\text{s}$ (Delay: $18.50\text{s}$)
   - **Peak Probability**: $0.9991$
4. **`PN12_sz03`** (Recording: `PN12/PN12-3.edf`, Duration: $96.0\text{s}$, Start: $772.0\text{s}$):
   - **Detected**: YES at $t=800.0\text{s}$ (Delay: $28.00\text{s}$)
   - **Peak Probability**: $0.9998$

## 2. False Alarm Analysis
- **Total False Alarm Episodes**: **0 episodes (0.00 FA/day)**
- **Raw False Positive Windows**: **5 windows** out of 3,716 negative windows (Window Specificity: **99.86%**)
- **Mechanism of Elimination**: The 5 false positive windows occurred as isolated, single 5-second transient spikes. The frozen 3-window moving average smoothing ($L=3$) and minimum alarm duration filter ($5.0\text{s} \equiv 3$ consecutive windows) successfully suppressed all 5 isolated noise transients without triggering a single clinical false alarm episode.