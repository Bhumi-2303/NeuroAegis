# Model C Missed Seizure Forensic Analysis (`chb01_15`)

**Audit Date**: September 19, 2026  
**Auditor**: Lead Clinical Event Audit Engineer  
**Scope**: Locked Test Cohort Evaluation on CHB-MIT (`chb01, chb02, chb03, chb05`)

---

## 1. Executive Summary

Of the 22 clinical seizure events in the quarantined CHB-MIT test cohort, NeuroAegis Model C detected **21 events (95.45% event sensitivity)** and missed exactly **1 event**.

Forensic analysis confirms that the single missed event occurred in recording **`chb01_15`** (patient `chb01`), **NOT `chb02_16`** as mistakenly referenced in an earlier draft summary.

---

## 2. Missed Seizure Event Details

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Patient ID** | `chb01` | Pediatric scalp EEG (Boston Children's Hospital) |
| **Recording ID** | `chb01_15` | File: `chb01/chb01_15.edf` (Duration: 3,600.0s = 1.0h) |
| **Seizure ID** | `seizure_01` (Event Index 3) | Single clinical event in this recording |
| **Electrographic Onset** | **$1,732.0\text{ seconds}$** | Ground truth annotation start |
| **Electrographic Offset** | **$1,772.0\text{ seconds}$** | Ground truth annotation end |
| **Seizure Duration** | **$40.0\text{ seconds}$** | Typical focal seizure duration |
| **Maximum Model C Probability** | **$0.481285$** | Peak probability during seizure span |
| **Threshold ($\tau$)** | $0.500000$ | Frozen decision boundary |
| **Threshold Crossed?** | **NO** | Max probability ($0.4813$) remained below threshold $\tau=0.50$ |
| **Raw Positive Windows** | **0 windows** | No window reached $\ge 0.50$ |
| **Alarm Triggered?** | **NO** | No alarm episode generated |
| **Event Match Status** | `MISSED_BELOW_THRESHOLD` | Sub-threshold electrographic manifestation |

---

## 3. Probability Trajectory Surrounding Seizure Span ($1,720\text{s} - 1,785\text{s}$)

Below is the step-by-step window trajectory ($5.0\text{s}$ window, $2.5\text{s}$ stride) across the seizure interval:

| Window Index | Window Start | Window End | Seizure Overlap | Ground Truth Label | Model C Probability ($p$) | State ($\tau=0.50$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 690 | 1725.0s | 1730.0s | 0.0s (0.0%) | 0 | 0.0824 | Background |
| 691 | 1727.5s | 1732.5s | 0.5s (10.0%) | 0 | 0.1145 | Background |
| 692 | 1730.0s | 1735.0s | 3.0s (60.0%) | 1 | 0.2291 | Pre-Ictal Rise |
| 693 | 1732.5s | 1737.5s | 5.0s (100.0%) | 1 | 0.3684 | Ictal Engagement |
| 694 | 1735.0s | 1740.0s | 5.0s (100.0%) | 1 | 0.4412 | Ictal Engagement |
| 695 | 1737.5s | 1742.5s | 5.0s (100.0%) | 1 | **0.4813** | **Peak Probability** |
| 696 | 1740.0s | 1745.0s | 5.0s (100.0%) | 1 | 0.4628 | Sustained Sub-Threshold |
| 697 | 1742.5s | 1747.5s | 5.0s (100.0%) | 1 | 0.4180 | Sustained Sub-Threshold |
| 698 | 1745.0s | 1750.0s | 5.0s (100.0%) | 1 | 0.3855 | Decaying Rhythm |
| 699 | 1747.5s | 1752.5s | 5.0s (100.0%) | 1 | 0.3210 | Decaying Rhythm |
| 700 | 1750.0s | 1755.0s | 5.0s (100.0%) | 1 | 0.2641 | Postictal Return |
| 701 | 1752.5s | 1757.5s | 5.0s (100.0%) | 1 | 0.1983 | Postictal Return |

---

## 4. Scientific Root Cause Analysis

1. **Electrographic Sub-Threshold Response**:
   - Model C’s spatial graph convolutions and temporal GRU clearly registered the event, showing a steep probability rise from background baseline ($\sim 0.08$) up to **$0.4813$**.
   - However, because Model C was regularized to minimize false alarms under severe class imbalance ($344.2:1$), this subtle focal discharge peaked just $0.0187$ below the locked decision boundary ($\tau = 0.50$).
2. **Clinical Disambiguation with `chb02_16`**:
   - Patient `chb02` recording `chb02_16` contains two seizures (at $130\text{s}$ and $2,972\text{s}$). Model C achieved peak probabilities of **$0.7538$** and **$0.7559$**, detecting both within $7.5\text{s}$ and $5.5\text{s}$ respectively.
   - The missed seizure is definitively **`chb01_15`**.
