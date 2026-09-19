# NeuroAegis — Evaluation Protocol V1.0 Metric Definitions Audit

**Document**: Mathematical & Clinical Formalization of Metrics (A through T)  
**Standard**: NeuroAegis Evaluation Protocol V1.0  
**Scope**: All continuous scalp EEG seizure detection evaluations.

---

## 1. Ground Truth & Windowing Definitions

### A. Dataset Split
- **Patient Isolation**: Patients are partitioned strictly into disjoint sets: $\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset$, $\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$, $\mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$.
- **CHB-MIT Quarantined Test Split**: $\mathcal{P}_{\text{test}} = \{\text{chb01}, \text{chb02}, \text{chb03}, \text{chb05}\}$ ($155\text{ EDF recordings}$, $152.8231\text{ hours}$).

### B. Window Definition
- **Window Length ($W$)**: $5.0\text{ seconds}$ ($1,280\text{ samples}$ at $f_s = 256\text{ Hz}$).
- **Window Stride ($S$)**: $2.5\text{ seconds}$ ($640\text{ samples}$ at $f_s = 256\text{ Hz}$).
- **Overlap**: $50\%$ temporal overlap between consecutive windows.

### C. Window Labels (`label_50pct_overlap`)
A window $w_i = [t_{\text{start}}^{(i)}, t_{\text{end}}^{(i)}]$ where $t_{\text{end}}^{(i)} - t_{\text{start}}^{(i)} = 5.0\text{s}$ is assigned binary label $y_i \in \{0, 1\}$:
$$y_i = \begin{cases} 1 & \text{if } \frac{\text{Overlap}(w_i, \mathcal{E}_{\text{seizure}})}{5.0\text{s}} \ge 0.50 \\ 0 & \text{otherwise} \end{cases}$$
where $\mathcal{E}_{\text{seizure}} = \bigcup_k [S_k^{\text{start}}, S_k^{\text{end}}]$ denotes the set of true clinical seizure intervals.

### D. Decision Threshold ($\tau$)
- **CHB-MIT Test Evaluation**: $\tau = 0.50$ (strictly frozen from validation set selection).
- **Siena Zero-Shot (Exp 6A)**: $\tau = 0.50$ (unmodified source threshold).
- **Siena Calibration (Exp 6B)**: $\tau^* = 0.50$ (selected on `PN00`, applied to held-out `PN12`).

---

## 2. Window-Level Classification Metrics

Given true binary labels $y_i \in \{0, 1\}$, continuous predicted probabilities $p_i \in [0.0, 1.0]$, and threshold $\tau = 0.50$, binary predictions are $\hat{y}_i = \mathbb{I}(p_i \ge \tau)$.

### E. Window Confusion Matrix Elements
- **True Positives ($TP$)**: $\sum_{i=1}^N \mathbb{I}(y_i = 1 \land \hat{y}_i = 1)$
- **False Positives ($FP$)**: $\sum_{i=1}^N \mathbb{I}(y_i = 0 \land \hat{y}_i = 1)$
- **True Negatives ($TN$)**: $\sum_{i=1}^N \mathbb{I}(y_i = 0 \land \hat{y}_i = 0)$
- **False Negatives ($FN$)**: $\sum_{i=1}^N \mathbb{I}(y_i = 1 \land \hat{y}_i = 0)$

### N. Precision (Positive Predictive Value)
$$\text{Precision} = \frac{TP}{TP + FP}$$

### O. Window Sensitivity (Recall / True Positive Rate)
$$\text{Window Sensitivity} = \frac{TP}{TP + FN}$$

### P. Window Specificity (True Negative Rate)
$$\text{Window Specificity} = \frac{TN}{TN + FP}$$

### Q. Window F1 Score
$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Sensitivity}}{\text{Precision} + \text{Sensitivity}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

### R. Balanced Accuracy
$$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$

### S. Area Under the ROC Curve (AUROC)
Calculated via trapezoidal integration over all possible threshold values:
$$\text{AUROC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

### T. Area Under the Precision-Recall Curve (AUPRC)
Calculated as average precision score using unthresholded probabilities:
$$\text{AUPRC} = \sum_{k=1}^K (R_k - R_{k-1}) P_k$$

---

## 3. Post-Processing & Alarm Episode Construction

### G. Alarm Episode Formation Protocol
Given the binary sequence $\hat{y}_i \in \{0, 1\}$ for an EDF recording:
1. **Temporal Majority Smoothing**: Sliding window majority filter of size $M=3$ windows ($7.5\text{s}$ span, pad $= 1$):
   $$\tilde{y}_i = \mathbb{I}\left(\sum_{j=i-1}^{i+1} \hat{y}_j \ge 2\right)$$
2. **Interval Extraction**: Contiguous sequences where $\tilde{y}_i = 1$ define candidate intervals $[t_{\text{start}}, t_{\text{end}}]$ with $t_{\text{start}} = i_{\text{start}} \times S$ and $t_{\text{end}} = i_{\text{end}} \times S + W$.
3. **Alarm Merging**: If the temporal gap between consecutive alarms $A_k$ and $A_{k+1}$ satisfies:
   $$t_{\text{start}}^{(k+1)} - t_{\text{end}}^{(k)} \le \Delta_{\text{merge}} = 15.0\text{ seconds}$$
   they are merged into a single alarm $[t_{\text{start}}^{(k)}, t_{\text{end}}^{(k+1)}]$.
4. **Duration Filtering**: Alarms shorter than minimum duration ($d < 5.0\text{ seconds}$) are discarded.

---

## 4. Event-Level Evaluation Metrics

### F. Event Matching Protocol
Let $\mathcal{E}_j = [S_j^{\text{start}}, S_j^{\text{end}}]$ be an annotated seizure event.  
An event $\mathcal{E}_j$ is **DETECTED** if there exists at least one post-processed alarm episode $\mathcal{A}_m = [A_m^{\text{start}}, A_m^{\text{end}}]$ such that:
1. $\mathcal{A}_m \cap \mathcal{E}_j \ne \emptyset$ (i.e., $A_m^{\text{end}} > S_j^{\text{start}}$ and $A_m^{\text{start}} < S_j^{\text{end}}$).
2. Detection delay $\delta_j = A_m^{\text{start}} - S_j^{\text{start}} \le \Delta_{\text{allowed}} = 30.0\text{ seconds}$.

- **Event Sensitivity**:
  $$\text{Event Sensitivity} = \frac{N_{\text{detected}}}{N_{\text{total}}}$$

### J. Detection Onset
$$\text{Alarm Onset Timestamp} = A_m^{\text{start}} = i_{\text{first\_detect}} \times 2.5\text{s}$$

### K. Detection Delay
$$\text{Onset-Referenced Detection Delay} = \max(0.0, A_m^{\text{start}} - S_j^{\text{start}})$$
- Primary Metric: **Mean Onset-Referenced Detection Delay** ($5.57\text{s}$ for Model C).
- Completion-Based Reference: $\text{Completion Delay} = A_m^{\text{end}} - S_j^{\text{start}} = \text{Onset Delay} + 5.0\text{s}$ ($10.57\text{s}$ for Model C).

---

## 5. False Alarm & Monitoring Rate Metrics

### I. Monitoring Duration ($H$)
$$H = \frac{\sum_{r \in \mathcal{R}_{\text{test}}} \text{EDF\_Duration\_Sec}(r)}{3600.0} = 152.8231\text{ hours}$$

### L. Raw False-Positive Window Rate
$$R_{\text{raw\_FP}} = \frac{\sum_{i=1}^N \mathbb{I}(y_i = 0 \land \hat{y}_i = 1)}{H} \times 24.0 = \frac{FP}{H} \times 24.0 \quad [\text{Raw FP windows / 24h}]$$
- For Model C: $399 / 152.8231 \times 24.0 = \mathbf{62.66\text{ Raw FP windows / 24h}}$.

### M. Clinical False-Alarm Episode Rate
Any constructed alarm episode $\mathcal{A}_m$ that does not overlap any true seizure $\mathcal{E}_j$ is an **Unmatched Clinical False Alarm Episode**.
$$R_{\text{clinical\_FA}} = \frac{N_{\text{unmatched\_alarm\_episodes}}}{H} \times 24.0 \quad [\text{Clinical FA episodes / 24h}]$$
- For Model C: $47 / 152.8231 \times 24.0 = \mathbf{7.38\text{ Clinical FA episodes / 24h}}$.

> [!CAUTION]
> **Reporting Rule**: Raw FP window rates and clinical false-alarm episode rates measure fundamentally distinct quantities. They must never be combined, plotted on identical axes, or interchangeably termed "false alarms/day".
