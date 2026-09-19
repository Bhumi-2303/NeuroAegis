# NeuroAegis — Final Cross-Experiment Metric Discrepancy Reconciliation Log

**Document Version**: 1.0 (Frozen)  
**Date**: 2026-09-19  
**Scope**: Forensic resolution of all mathematical, methodological, and terminology discrepancies identified during the cross-experiment audit.

---

## 1. Discrepancy 1: Model C False Alarm Rate (62.66 vs. 12.56 vs. 7.38 / 24h)

### A. Context & Symptoms
Historical drafts and experiment summaries reported three different numbers for Model C false alarms on the 152.82h CHB-MIT test set:
- **62.66 / 24h**
- **12.56 / 24h**
- **7.38 / 24h** (or 7.22 / 24h in preliminary harness runs)

### B. Root Cause Analysis
1. **Raw Unclustered FP Windows (62.66 / 24h)**:
   - On the 219,909 test windows, Model C produced exactly **399 false positive windows** ($y=0, \hat{y}=1$).
   - Across $152.8231	ext{ hours}$:
     $$	ext{Raw FP / 24h} = rac{399}{152.8231} 	imes 24 = 62.6601 pprox 62.66	ext{ FP / 24h}$$
   - This measures raw computational error at the 2.5-second stride level without clinical clustering.

2. **Historical Contiguous Run Clustered Episodes (12.56 / 24h)**:
   - Early scripts merged contiguous positive windows ($y_{	ext{pred}}=1$) without median/majority filtering, yielding **80 contiguous alarm blocks**:
     $$	ext{Historical Clustered FA / 24h} = rac{80}{152.8231} 	imes 24 = 12.5635 pprox 12.56	ext{ FA / 24h}$$

3. **Protocol V1.0 Clinical Alarm Episodes (7.38 / 24h)**:
   - Under the authoritative frozen Protocol V1.0 harness:
     - 3-window majority filtering removes 1-window transient spikes.
     - Inter-alarm gaps $< 15.0	ext{s}$ are merged into single clinical episodes.
     - Alarms $< 5.0	ext{s}$ are discarded as sub-threshold chatter.
   - This produces exactly **47 discrete clinical false alarm episodes**:
     $$	ext{Protocol V1 Clinical FA / 24h} = rac{47}{152.8231} 	imes 24 = 7.3811 pprox 7.38	ext{ FA / 24h}$$

### C. Resolution & Reporting Rule
- **Rule**: Research manuscripts must report **both**:
  1. **Clinical False Alarm Episodes**: `7.38 FA / 24h` (47 episodes across 152.82h)
  2. **Raw False Positive Windows**: `62.66 FP / 24h` (399 windows across 219,272 negative windows, Specificity = `99.82%`)
- **Status**: `RECONCILED & FROZEN`

---

## 2. Discrepancy 2: Model C Detection Delay (5.57s vs. 6.05s vs. 10.57s)

### A. Context & Symptoms
Different evaluation logs reported detection latencies for Model C of **5.57 seconds**, **6.05 seconds**, and **10.57 seconds**.

### B. Root Cause Analysis
1. **Raw Window Onset Delay (5.57s)**:
   - Calculated from the start timestamp of the first raw window where $p \ge 0.50$:
     $$	ext{Delay}_{	ext{raw}} = T_{	ext{window\_start}} - T_{	ext{seizure\_onset}}$$
   - Across the 21 detected seizures, the mean raw onset delay is **5.57 seconds** (median **4.00s**).

2. **Protocol V1 Alarm Episode Onset Delay (6.05s)**:
   - Protocol V1 applies a 3-window majority filter before raising an alarm. For 20 of the 21 detected seizures, the raw onset window coincides with the start of the majority-filtered episode.
   - In Seizure 4 (`chb01_04`), an isolated positive window appeared at $+0.5	ext{s}$, followed by a brief 1-window drop, before continuous sustained firing at $+10.5	ext{s}$. The majority filter suppresses the single-window transient, locking alarm onset at $+10.5	ext{s}$ ($+10.0	ext{s}$ shift on this event).
   - Mean over 21 events: $rac{20 	imes 5.57 + 10.0}{21} = 6.0476 pprox \mathbf{6.05	ext{ seconds}}$.

3. **Window Completion / Buffer Delay (10.57s)**:
   - Measured from the completion of the 5.0-second EEG window ($T_{	ext{window\_end}} - T_{	ext{seizure\_onset}}$):
     $$	ext{Delay}_{	ext{end}} = 5.57	ext{s} + 5.00	ext{s} = \mathbf{10.57	ext{ seconds}}$$

### C. Resolution & Reporting Rule
- **Authoritative Primary Latency**: **`6.05 seconds`** (Protocol V1 Alarm Episode Onset)
- **Physical Earliest Detection Latency**: **`5.57 seconds`** (Raw Window Onset)
- **Buffer-Complete Latency**: **`10.57 seconds`** (5.0s window buffer offset)
- **Status**: `RECONCILED & FROZEN`

---

## 3. Discrepancy 3: Ground Truth Seizure Audit & Single Missed Seizure Identity

### A. Context & Symptoms
An early working draft mentioned `chb02_16` as the missed seizure.

### B. Forensic Verification
- A full trace of all 22 ground truth seizure events on the test cohort (`chb01, chb02, chb03, chb05`) reveals:
  - `chb02_16` has 2 annotated seizures (onset 130s and 2966s). Model C detects **both** seizures with high probabilities ($p=0.754$ and $p=0.756$).
  - The actual single missed seizure is **`chb01_15`** (onset 1,732s, end 1,772s; duration 40.0s).
  - During `chb01_15`, Model C predictions peaked at **$p = 0.4813$** (just below the fixed threshold $	au=0.50$).
- Total detected seizures: **21 of 22 (95.45% event sensitivity)**.

### C. Resolution
- The missed event is definitively documented as **`chb01_15`** (focal seizure, peak $p=0.4813$).
- **Status**: `VERIFIED`

---

## 4. Discrepancy 4: BENDR Foundation Model Representation Collapse & FA Paradox

### A. Context & Symptoms
The pretrained BENDR Biosignal Transformer pilot reported `AUROC = 0.58518`, `AUPRC = 0.00460`, `Window Specificity = 0.00%`, but `FA Episodes / 24h = 0.00`.

### B. Root Cause Analysis
1. **Representation Collapse**: All 219,909 test window probability predictions generated by the pilot adapter collapsed into a narrow band:
   $$p \in [0.212751, 0.212758], \quad \mu = 0.212754, \quad \sigma = 0.000001$$
2. **Threshold Artifact**: At the validation-selected threshold $	au=0.10$, every single window satisfies $p \ge 0.10$, producing an all-ones binary stream:
   - `Window Sensitivity = 100.00%` (637/637)
   - `Window Specificity = 0.00%` (0/219,272)
   - `Raw FP Windows = 219,272` (34,435.43 FP/24h)
3. **Episode Clustering Paradox**: The event-matching harness merged the unbroken all-ones sequence into **1 continuous alarm episode** spanning 152.82 hours. Because this continuous alarm overlaps all 22 true seizures, it matches them all and leaves $1 - 1 = 0$ unmatched false alarm episodes.

### C. Resolution
- BENDR is formally classified as a **Collapsed Alert State / Continuous Alert Failure**.
- Its clinical false alarm rate must NOT be reported as 0.00 FA/day without explicitly displaying the 219,272 raw FP windows (34,435.43 FP/day) and 0.00% specificity.
- **Status**: `RECONCILED & FLAGGED`

---

## 5. Discrepancy 5: Cross-Domain Siena Cohort Scope

### A. Context & Symptoms
Some early notes loosely referred to "Siena dataset validation" without quantifying the evaluated cohort.

### B. Forensic Verification
- The local repository shard contains recordings for **2 patients** (`PN00`, `PN12`), encompassing 6 EDF recordings, 2.67 hours of EEG, and 4 clinical seizure events.
- Frozen Model C evaluated on this shard achieves:
  - `AUROC = 0.89934`
  - `AUPRC = 0.70284`
  - `Event Sensitivity = 4/4 (100.00%)`
  - `Raw FP Windows / 24h = 0.00`
  - `Clinical FA Episodes / 24h = 0.00`
  - `Mean Onset Delay = 0.75 s`

### C. Resolution
- Must be explicitly labeled as **Limited External Zero-Shot Siena Subset (Feasibility Analysis: $N=2$ patients, 4 seizures, 2.67 hours)** to prevent overgeneralization claims.
- **Status**: `VERIFIED & PROPERLY QUALIFIED`

---

## 6. Discrepancy 6: Historical Temporal and EEG-Specific Latencies & FP Rates

### A. Symptoms
Previous reports for Suite 1 (Temporal) and Suite 2 (EEG-Specific DL) reported latencies ~5s higher and false alarm rates 10x-50x higher than Model C.

### B. Root Cause
1. Historical reports used **window completion delay** ($T_{	ext{window\_end}} - T_{	ext{seizure\_onset}}$), which includes the +5.0s window buffer. Under onset-referenced delay ($T_{	ext{window\_start}} - T_{	ext{seizure\_onset}}$), all values shift downward by exactly 5.0 seconds.
2. Historical reports listed **raw window false positives** (e.g., 40,748 FP windows for EEGNet = 6,399.24 FP/24h). Under the Protocol V1 clinical episode harness, 3-window majority filtering and 15s merging cluster these into **153.28 clinical FA episodes / 24h**.

### C. Status
- `RECONCILED & FULLY ALIGNED IN PROTOCOL V1.0 MASTER SCOREBOARD`
