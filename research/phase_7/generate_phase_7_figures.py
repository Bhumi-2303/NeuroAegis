"""
NeuroAegis Phase 7: Publication Figure Generation Engine
Generates all 16 required publication figures at 300 DPI into research/phase_7/figures/.

Figure Inventory:
1.  fig01_model_architecture_comparison.png: Architecture comparison schema (A vs B vs C)
2.  fig02_event_sensitivity_across_models.png: Primary clinical metric: Event Sensitivity
3.  fig03_auprc_across_models.png: AUPRC progression across architectural stages
4.  fig04_auroc_across_models.png: AUROC progression across architectural stages
5.  fig05_f1_across_models.png: Window-level F1 score progression
6.  fig06_false_alarms_across_models.png: False Alarm Rate (FA/24h) dramatic reduction
7.  fig07_detection_delay_across_models.png: Clinical Onset Detection Delay
8.  fig08_patient_wise_event_sensitivity.png: Patient x Model Event Sensitivity consistency
9.  fig09_patient_wise_f1.png: Patient x Model F1 score consistency
10. fig10_patient_wise_false_alarms.png: Patient x Model False Alarm Rates
11. fig11_detection_delay_distribution.png: Distribution of detection delay across 22 test events
12. fig12_ecdf_detection_delay.png: Empirical Cumulative Distribution Function of detection delay
13. fig13_complexity_vs_performance.png: Computational Efficiency: Parameters vs F1 and FA
14. fig14_threshold_sensitivity_validation.png: Validation-set diagnostic decision threshold sweep
15. fig15_calibration_reliability_curve.png: Reliability diagrams and calibration curves
16. fig16_ablation_summary.png: Multi-metric radar / waterfall ablation summary
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.gridspec as gridspec

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
PHASE7_DIR = os.path.join(BASE_DIR, "research/phase_7")
RESULTS_DIR = os.path.join(PHASE7_DIR, "results")
FIGURES_DIR = os.path.join(PHASE7_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

# Color palette
COLOR_A = "#4A5568"   # Slate Grey (Model A)
COLOR_B = "#ED8936"   # Amber/Orange (Model B)
COLOR_C = "#2B6CB0"   # Deep Navy Blue (Model C)
COLOR_ACCENT = "#38A169" # Forest Green

def generate_all_figures():
    print("=" * 80)
    print("NEUROAEGIS PHASE 7: PUBLICATION FIGURE GENERATION (16 FIGURES @ 300 DPI)")
    print("=" * 80)

    # Load result data
    df_comp = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    df_ablation = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"))
    df_pat = pd.read_csv(os.path.join(RESULTS_DIR, "patient_level_results.csv"))
    df_events = pd.read_csv(os.path.join(RESULTS_DIR, "event_level_results.csv"))
    df_fa = pd.read_csv(os.path.join(RESULTS_DIR, "false_alarm_results.csv"))
    df_delay = pd.read_csv(os.path.join(RESULTS_DIR, "detection_delay_results.csv"))
    df_boot = pd.read_csv(os.path.join(RESULTS_DIR, "bootstrap_results.csv"))
    df_cal = pd.read_csv(os.path.join(RESULTS_DIR, "calibration_results.csv"))
    df_thresh = pd.read_csv(os.path.join(RESULTS_DIR, "threshold_validation.csv"))

    # -------------------------------------------------------------
    # Fig 1: Model Architecture Comparison Diagram
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.95, "NeuroAegis Architectural Evolution & Component Ablation", 
            ha="center", va="top", fontsize=15, fontweight="bold", color="#1A202C")
    ax.text(0.5, 0.89, "Comparison of Architectural Stages on Untouched CHB-MIT Test Cohort (152.82 Hours)", 
            ha="center", va="top", fontsize=11, color="#4A5568", style="italic")

    # Draw 3 architecture columns
    cols = [
        ("Model A: 1D CNN Baseline", "Phase 3 (Temporal Only)", "173,601 Params", 
         ["4-Stage 1D Temporal Conv", "Receptive Field: ~5.0s", "GELU + MaxPool + Dropout", "Global AvgPool -> Linear Head", "No Spatial Graph", "No Temporal Recurrence"],
         "Event Sens: 100.0%*\nFA/24h: 1,946.56\nAUPRC: 0.0415\nF1 Score: 0.0155", 0.05, COLOR_A),
        ("Model B: CNN + Spatial GNN", "Phase 4A-C (Spatio-Temporal)", "52,497 Params (-69.8%)", 
         ["Multi-Channel 1D CNN Front-End", "Spatial Graph GNN (theta=0.30)", "Pearson Correlation Adjacency", "Graph Conv Layer (23 Bipolar Nodes)", "Static Spatial Smoothing", "No Temporal Recurrence"],
         "Event Sens: 27.27%\nFA/24h: 200.39 (-89.7%)\nAUPRC: 0.0049\nF1 Score: 0.0278", 0.38, COLOR_B),
        ("Model C: CNN + GNN + GRU", "Phase 4B (NeuroAegis Final)", "91,858 Params (+75.0% vs B)", 
         ["Frozen 1D CNN Front-End", "Frozen Spatial GNN (theta=0.30)", "Causal Unidirectional GRU", "Sequence Length L=8 (22.5s)", "Hidden Dim: 64, Dropout: 0.2", "Autoregressive Ictal Dynamics"],
         "Event Sens: 95.45%\nFA/24h: 62.66 (-96.8%)\nAUPRC: 0.8068 (16.4x)\nF1 Score: 0.6803 (43.8x)", 0.71, COLOR_C)
    ]

    for title, subtitle, param_str, items, metric_summary, x_pos, col_color in cols:
        # Outer box
        rect = FancyBboxPatch((x_pos, 0.08), 0.26, 0.76, boxstyle="round,pad=0.02",
                              edgecolor=col_color, facecolor="#F7FAFC", linewidth=2.0)
        ax.add_patch(rect)
        
        # Header banner
        banner = FancyBboxPatch((x_pos, 0.72), 0.26, 0.12, boxstyle="round,pad=0.01",
                                edgecolor=col_color, facecolor=col_color, linewidth=1.5)
        ax.add_patch(banner)
        ax.text(x_pos + 0.13, 0.80, title, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x_pos + 0.13, 0.75, f"{subtitle} | {param_str}", ha="center", va="center", color="#E2E8F0", fontsize=8.5)

        # Architectural features
        y_text = 0.68
        for item in items:
            ax.text(x_pos + 0.02, y_text, f"* {item}", va="top", fontsize=9, color="#2D3748")
            y_text -= 0.052

        # Divider
        ax.plot([x_pos + 0.02, x_pos + 0.24], [0.34, 0.34], color="#CBD5E0", lw=1.2, linestyle="--")

        # Metrics box
        ax.text(x_pos + 0.03, 0.31, "Test Performance:", fontsize=9.5, fontweight="bold", color=col_color)
        ax.text(x_pos + 0.03, 0.27, metric_summary, fontsize=9, color="#1A202C", va="top", linespacing=1.35)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig01_model_architecture_comparison.png"))
    plt.close()
    print("  -> Generated fig01_model_architecture_comparison.png")

    # -------------------------------------------------------------
    # Fig 2: Event Sensitivity Across Models
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    models = ["Model A\n(1D CNN)", "Model B\n(CNN+GNN)", "Model C\n(CNN+GNN+GRU)"]
    # Nominal vs Strict
    ev_sens_nominal = [100.0, 27.27, 95.45]
    ev_sens_strict = [54.55, 27.27, 95.45]
    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, ev_sens_nominal, width, label="Nominal / Reported Event Sens (%)", color=[COLOR_A, COLOR_B, COLOR_C], alpha=0.6, edgecolor="black")
    rects2 = ax.bar(x + width/2, ev_sens_strict, width, label="Strict Event-Disambiguated Sens (%)", color=[COLOR_A, COLOR_B, COLOR_C], alpha=1.0, edgecolor="black")

    ax.set_ylabel("Event-Level Sensitivity (%)", fontweight="bold")
    ax.set_title("Figure 2: Clinical Seizure Event Sensitivity Across Architectural Models\n(Untouched CHB-MIT Test Partition: 22 Seizures Across 4 Patients)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(90, color="#E53E3E", linestyle="--", alpha=0.7, label="Clinical Target Benchmark (90%)")
    ax.legend(loc="lower right", frameon=True)

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig02_event_sensitivity_across_models.png"))
    plt.close()
    print("  -> Generated fig02_event_sensitivity_across_models.png")

    # -------------------------------------------------------------
    # Fig 3: AUPRC Across Models
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    auprcs = [0.04148, 0.00492, 0.80681]
    prev = 637 / 219909  # 0.002896
    colors = [COLOR_A, COLOR_B, COLOR_C]

    bars = ax.bar(models, auprcs, color=colors, edgecolor="black", width=0.55, linewidth=1.2)
    ax.axhline(prev, color="red", linestyle=":", linewidth=1.5, label=f"Random Prevalence Baseline ({prev*100:.2f}%)")
    ax.set_ylabel("Area Under Precision-Recall Curve (AUPRC)", fontweight="bold")
    ax.set_title("Figure 3: Test AUPRC Progression Across Architectural Stages\n(High-Imbalance Continuous Monitoring: 344.23:1)", fontweight="bold")
    ax.set_ylim(0, 0.95)

    for bar, val in zip(bars, auprcs):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.02, f"{val:.4f}\n({val/prev:.1f}x baseline)", 
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.annotate("Temporal GRU Recurrence:\n+0.8019 vs Spatial GNN\n+0.7653 vs 1D CNN", 
                xy=(2, 0.8068), xytext=(1.2, 0.70),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF8FF", edgecolor=COLOR_C),
                fontsize=9.5, fontweight="bold")

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig03_auprc_across_models.png"))
    plt.close()
    print("  -> Generated fig03_auprc_across_models.png")

    # -------------------------------------------------------------
    # Fig 4: AUROC Across Models
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    aurocs = [0.36389, 0.19431, 0.98970]

    bars = ax.bar(models, aurocs, color=colors, edgecolor="black", width=0.55, linewidth=1.2)
    ax.axhline(0.50, color="gray", linestyle="--", linewidth=1.2, label="Chance Discrimination (0.50)")
    ax.set_ylabel("Area Under ROC Curve (AUROC)", fontweight="bold")
    ax.set_title("Figure 4: Test AUROC Across Architectural Models\n(Patient-Independent Continuous EEG: 219,909 Windows)", fontweight="bold")
    ax.set_ylim(0, 1.15)

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.02, f"{val:.4f}", 
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.annotate("Near-Perfect Patient-Independent\nSeparation: AUROC = 0.9897", 
                xy=(2, 0.9897), xytext=(1.1, 0.85),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF8FF", edgecolor=COLOR_C),
                fontsize=9.5, fontweight="bold")

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig04_auroc_across_models.png"))
    plt.close()
    print("  -> Generated fig04_auroc_across_models.png")

    # -------------------------------------------------------------
    # Fig 5: F1 Across Models
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    f1s = [0.01553, 0.02784, 0.68025]

    bars = ax.bar(models, f1s, color=colors, edgecolor="black", width=0.55, linewidth=1.2)
    ax.set_ylabel("Window-Level F1 Score (tau = 0.50)", fontweight="bold")
    ax.set_title("Figure 5: Window-Level F1 Score Progression\n(Precision-Recall Harmonic Mean at Frozen tau=0.50)", fontweight="bold")
    ax.set_ylim(0, 0.85)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.02, f"{val:.4f}", 
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.annotate("43.8x F1 Increase over 1D CNN\n24.4x F1 Increase over CNN+GNN", 
                xy=(2, 0.68025), xytext=(0.95, 0.55),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF8FF", edgecolor=COLOR_C),
                fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig05_f1_across_models.png"))
    plt.close()
    print("  -> Generated fig05_f1_across_models.png")

    # -------------------------------------------------------------
    # Fig 6: False Alarms Across Models (FA/24h)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    fas = [1946.56, 200.39, 62.66]

    bars = ax.bar(models, fas, color=colors, edgecolor="black", width=0.55, linewidth=1.2)
    ax.set_ylabel("False Alarms per 24 Hours (FA/24h)", fontweight="bold")
    ax.set_title("Figure 6: Dramatic Reduction in False Alarm Frequency\n(Continuous Monitoring: 152.82 Hours)", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(10, 5000)

    # Annotate values and percent reductions
    ax.text(bars[0].get_x() + bars[0].get_width()/2., fas[0] * 1.15, f"{fas[0]:.1f} / day\n(12,395 FP)", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(bars[1].get_x() + bars[1].get_width()/2., fas[1] * 1.15, f"{fas[1]:.1f} / day\n(-89.7% vs A)", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(bars[2].get_x() + bars[2].get_width()/2., fas[2] * 1.15, f"{fas[2]:.1f} / day\n(-96.8% vs A)", ha="center", fontsize=9.5, fontweight="bold")

    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig06_false_alarms_across_models.png"))
    plt.close()
    print("  -> Generated fig06_false_alarms_across_models.png")

    # -------------------------------------------------------------
    # Fig 7: Detection Delay Across Models
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    delays = [9.58, 7.08, 10.57]
    det_counts = ["12/22 (54.5%)", "6/22 (27.3%)", "21/22 (95.5%)"]

    bars = ax.bar(models, delays, color=colors, edgecolor="black", width=0.55, linewidth=1.2)
    ax.axhline(10.0, color="#E53E3E", linestyle="--", linewidth=1.5, label="Clinical Target Window (10.0s)")
    ax.set_ylabel("Mean Detection Delay (seconds)", fontweight="bold")
    ax.set_title("Figure 7: Clinical Seizure Onset Detection Latency\n(Evaluated on Successfully Detected Seizures)", fontweight="bold")
    ax.set_ylim(0, 15)

    for bar, val, det in zip(bars, delays, det_counts):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.3, f"{val:.2f} s\n(Events: {det})", 
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig07_detection_delay_across_models.png"))
    plt.close()
    print("  -> Generated fig07_detection_delay_across_models.png")

    # -------------------------------------------------------------
    # Fig 8: Patient-Wise Event Sensitivity
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    patients = ["chb01\n(7 Seizures)", "chb02\n(3 Seizures)", "chb03\n(7 Seizures)", "chb05\n(5 Seizures)"]
    x = np.arange(len(patients))
    width = 0.25

    p_a_sens = [42.86, 66.67, 28.57, 100.0]
    p_b_sens = [0.0, 66.67, 0.0, 80.0]
    p_c_sens = [85.71, 100.0, 100.0, 100.0]

    rects1 = ax.bar(x - width, p_a_sens, width, label="Model A (1D CNN)", color=COLOR_A, edgecolor="black")
    rects2 = ax.bar(x, p_b_sens, width, label="Model B (CNN+GNN)", color=COLOR_B, edgecolor="black")
    rects3 = ax.bar(x + width, p_c_sens, width, label="Model C (CNN+GNN+GRU)", color=COLOR_C, edgecolor="black")

    ax.set_ylabel("Event Sensitivity (%)", fontweight="bold")
    ax.set_title("Figure 8: Patient-Wise Clinical Event Sensitivity Consistency\n(Evaluating Robustness Across Unseen Patients)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(patients, fontweight="bold")
    ax.set_ylim(0, 120)
    ax.legend(loc="upper right", frameon=True)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x() + rect.get_width()/2., h + 2, f"{h:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig08_patient_wise_event_sensitivity.png"))
    plt.close()
    print("  -> Generated fig08_patient_wise_event_sensitivity.png")

    # -------------------------------------------------------------
    # Fig 9: Patient-Wise F1 Scores
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    p_a_f1 = [0.1810, 0.0895, 0.0938, 0.0082]
    p_b_f1 = [0.0000, 0.4043, 0.0000, 0.0111]
    p_c_f1 = [0.8614, 0.7425, 0.6786, 0.5773]

    rects1 = ax.bar(x - width, p_a_f1, width, label="Model A (1D CNN)", color=COLOR_A, edgecolor="black")
    rects2 = ax.bar(x, p_b_f1, width, label="Model B (CNN+GNN)", color=COLOR_B, edgecolor="black")
    rects3 = ax.bar(x + width, p_c_f1, width, label="Model C (CNN+GNN+GRU)", color=COLOR_C, edgecolor="black")

    ax.set_ylabel("F1 Score", fontweight="bold")
    ax.set_title("Figure 9: Patient-Wise Window-Level F1 Score Consistency\n(Model C Outperforms in 4 of 4 Patients)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(patients, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=True)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x() + rect.get_width()/2., h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig09_patient_wise_f1.png"))
    plt.close()
    print("  -> Generated fig09_patient_wise_f1.png")

    # -------------------------------------------------------------
    # Fig 10: Patient-Wise False Alarms
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    p_a_fa = [12.43, 286.51, 12.63, 7342.86]
    p_b_fa = [4.14, 3.40, 36.00, 742.77]
    p_c_fa = [5.33, 23.84, 60.67, 159.38]

    rects1 = ax.bar(x - width, p_a_fa, width, label="Model A (1D CNN)", color=COLOR_A, edgecolor="black")
    rects2 = ax.bar(x, p_b_fa, width, label="Model B (CNN+GNN)", color=COLOR_B, edgecolor="black")
    rects3 = ax.bar(x + width, p_c_fa, width, label="Model C (CNN+GNN+GRU)", color=COLOR_C, edgecolor="black")

    ax.set_ylabel("False Alarms per 24 Hours (Log Scale)", fontweight="bold")
    ax.set_title("Figure 10: Patient-Wise False Alarm Frequency\n(Elimination of Outlier False Positive Bursts in chb05)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(patients, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1, 15000)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig10_patient_wise_false_alarms.png"))
    plt.close()
    print("  -> Generated fig10_patient_wise_false_alarms.png")

    # -------------------------------------------------------------
    # Fig 11: Detection Delay Distribution for Model C
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    delays_c = df_events[df_events["model_c_detected"]]["model_c_delay_sec"].dropna().values

    # Histogram
    ax1.hist(delays_c, bins=np.arange(0, 35, 2.5), color=COLOR_C, edgecolor="black", alpha=0.8)
    ax1.axvline(np.median(delays_c), color="#E53E3E", linestyle="--", linewidth=1.5, label=f"Median ({np.median(delays_c):.1f}s)")
    ax1.axvline(np.mean(delays_c), color="#DD6B20", linestyle=":", linewidth=1.5, label=f"Mean ({np.mean(delays_c):.1f}s)")
    ax1.set_xlabel("Detection Delay (seconds)", fontweight="bold")
    ax1.set_ylabel("Number of Seizure Events", fontweight="bold")
    ax1.set_title("Distribution of Detection Latency (N=21)", fontweight="bold")
    ax1.legend(loc="upper right")

    # Boxplot comparison
    delays_all = [
        df_events[df_events["model_a_detected"]]["model_a_delay_sec"].dropna().values,
        df_events[df_events["model_b_detected"]]["model_b_delay_sec"].dropna().values,
        delays_c
    ]
    bp = ax2.boxplot(delays_all, patch_artist=True)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["Model A\n(N=12)", "Model B\n(N=6)", "Model C\n(N=21)"])
    for patch, col in zip(bp["boxes"], [COLOR_A, COLOR_B, COLOR_C]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set(color="red", linewidth=2.0)
    ax2.axhline(10.0, color="#E53E3E", linestyle="--", linewidth=1.2, label="10s Target Benchmark")
    ax2.set_ylabel("Detection Delay (seconds)", fontweight="bold")
    ax2.set_title("Cross-Model Latency Boxplots", fontweight="bold")
    ax2.legend(loc="upper right")

    plt.suptitle("Figure 11: Clinical Detection Latency Characteristics", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig11_detection_delay_distribution.png"))
    plt.close()
    print("  -> Generated fig11_detection_delay_distribution.png")

    # -------------------------------------------------------------
    # Fig 12: Empirical CDF of Detection Delay
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    # ECDF for Model C
    sorted_delays = np.sort(delays_c)
    ecdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)

    ax.step(sorted_delays, ecdf, where="post", color=COLOR_C, linewidth=2.5, label="Model C ECDF (21 Events)")
    ax.scatter(sorted_delays, ecdf, color=COLOR_C, s=35, zorder=5)

    # Reference lines
    ax.axvline(5.0, color="gray", linestyle=":", label="5.0s (1 Stride)")
    ax.axvline(10.0, color="#E53E3E", linestyle="--", label="10.0s Clinical Target")
    
    # Fractions
    f_5 = (sorted_delays <= 5.0).mean() * 100
    f_10 = (sorted_delays <= 10.0).mean() * 100
    ax.text(5.2, 0.25, f"{f_5:.1f}% <= 5.0s", color="gray", fontweight="bold")
    ax.text(10.2, 0.65, f"{f_10:.1f}% <= 10.0s", color="#E53E3E", fontweight="bold")

    ax.set_xlabel("Time from Seizure Onset to First Alarm (seconds)", fontweight="bold")
    ax.set_ylabel("Cumulative Detection Probability", fontweight="bold")
    ax.set_title("Figure 12: Empirical Cumulative Distribution Function (ECDF)\nof Detection Delay for Final Model C", fontweight="bold")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig12_ecdf_detection_delay.png"))
    plt.close()
    print("  -> Generated fig12_ecdf_detection_delay.png")

    # -------------------------------------------------------------
    # Fig 13: Computational Complexity vs Performance
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    params = [173601, 52497, 91858]
    f1s_plot = [0.01553, 0.02784, 0.68025]
    fa_plot = [1946.56, 200.39, 62.66]
    labels = ["Model A (CNN)", "Model B (CNN+GNN)", "Model C (CNN+GNN+GRU)"]

    for x_p, y_f, y_fa, col, lbl in zip(params, f1s_plot, fa_plot, colors, labels):
        ax1.scatter(x_p, y_f, s=180, color=col, edgecolors="black", linewidth=1.5, label=lbl, zorder=5)
        ax2.scatter(x_p, y_fa, s=180, color=col, edgecolors="black", linewidth=1.5, label=lbl, zorder=5)

    ax1.set_xlabel("Trainable Parameters", fontweight="bold")
    ax1.set_ylabel("Window-Level F1 Score", fontweight="bold")
    ax1.set_title("Parameters vs F1 Score", fontweight="bold")
    ax1.set_ylim(-0.05, 0.85)
    ax1.legend(loc="upper left")

    ax2.set_xlabel("Trainable Parameters", fontweight="bold")
    ax2.set_ylabel("False Alarms / 24h (Log Scale)", fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_title("Parameters vs False Alarm Rate", fontweight="bold")
    ax2.set_ylim(10, 5000)
    ax2.legend(loc="upper right")

    plt.suptitle("Figure 13: Model Parameter Efficiency vs Clinical Utility Frontier", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig13_complexity_vs_performance.png"))
    plt.close()
    print("  -> Generated fig13_complexity_vs_performance.png")

    # -------------------------------------------------------------
    # Fig 14: Threshold Sensitivity on Validation Set
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    t_vals = df_thresh["threshold"].values
    sens_vals = df_thresh["sensitivity"].values
    spec_vals = df_thresh["specificity"].values
    f1_vals = df_thresh["f1_score"].values
    prec_vals = df_thresh["precision"].values

    ax.plot(t_vals, sens_vals, marker="o", color="#E53E3E", linewidth=2.0, label="Window Sensitivity")
    ax.plot(t_vals, spec_vals, marker="s", color="#3182CE", linewidth=2.0, label="Window Specificity")
    ax.plot(t_vals, prec_vals, marker="^", color="#DD6B20", linewidth=2.0, label="Precision")
    ax.plot(t_vals, f1_vals, marker="d", color="#38A169", linewidth=2.5, label="F1 Score")

    ax.axvline(0.50, color="black", linestyle="--", linewidth=1.5, label="Frozen Test Threshold (tau = 0.50)")
    ax.set_xlabel("Decision Threshold (tau)", fontweight="bold")
    ax.set_ylabel("Metric Value", fontweight="bold")
    ax.set_title("Figure 14: Diagnostic Decision Threshold Sweep on Validation Cohort\n(CHB-MIT Val: 293,410 Windows, 203.82 Hours — Zero Test Leakage)", fontweight="bold")
    ax.set_xticks(t_vals)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center left", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig14_threshold_sensitivity_validation.png"))
    plt.close()
    print("  -> Generated fig14_threshold_sensitivity_validation.png")

    # -------------------------------------------------------------
    # Fig 15: Calibration Reliability Diagram
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    
    # Reliability curves
    cal_c = df_cal[df_cal["model_id"] == "Model C"]
    cal_b = df_cal[df_cal["model_id"] == "Model B"]
    cal_a = df_cal[df_cal["model_id"] == "Model A"]

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax1.plot(cal_a["mean_confidence"], cal_a["empirical_accuracy"], marker="o", color=COLOR_A, label=f"Model A (Brier={cal_a['brier_score'].iloc[0]:.4f})")
    ax1.plot(cal_b["mean_confidence"], cal_b["empirical_accuracy"], marker="s", color=COLOR_B, label=f"Model B (Brier={cal_b['brier_score'].iloc[0]:.4f})")
    ax1.plot(cal_c["mean_confidence"], cal_c["empirical_accuracy"], marker="^", color=COLOR_C, linewidth=2.2, label=f"Model C (Brier={cal_c['brier_score'].iloc[0]:.4f})")

    ax1.set_xlabel("Mean Predicted Probability (Confidence)", fontweight="bold")
    ax1.set_ylabel("Fraction of Positives (Empirical Accuracy)", fontweight="bold")
    ax1.set_title("Reliability Diagram Across Models", fontweight="bold")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="upper left")

    # Model C calibration error bars
    bar_x = np.arange(len(cal_c))
    ax2.bar(bar_x - 0.15, cal_c["mean_confidence"], width=0.3, label="Mean Confidence", color="#90CDF4", edgecolor="black")
    ax2.bar(bar_x + 0.15, cal_c["empirical_accuracy"], width=0.3, label="Empirical Accuracy", color=COLOR_C, edgecolor="black")
    ax2.set_xlabel("Probability Bin Index", fontweight="bold")
    ax2.set_ylabel("Probability Value", fontweight="bold")
    ax2.set_title(f"Model C Bin Confidence vs Accuracy\n(ECE = {cal_c['ece'].iloc[0]:.4f}, MCE = {cal_c['mce'].iloc[0]:.4f})", fontweight="bold")
    ax2.set_xticks(bar_x)
    ax2.set_xticklabels([f"[{r.bin_lower:.1f},{r.bin_upper:.1f}]" for _, r in cal_c.iterrows()], rotation=45, ha="right", fontsize=8)
    ax2.legend(loc="upper left")

    plt.suptitle("Figure 15: Probability Calibration & Reliability Analysis", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig15_calibration_reliability_curve.png"))
    plt.close()
    print("  -> Generated fig15_calibration_reliability_curve.png")

    # -------------------------------------------------------------
    # Fig 16: Multi-Metric Ablation Summary
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

    # Radar chart
    categories = ["Event Sensitivity", "AUPRC", "AUROC", "F1 Score", "Specificity", "Balanced Acc"]
    N_cat = len(categories)
    angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
    angles += angles[:1]

    # Normalized values in [0, 1]
    val_a = [0.5455, 0.0415, 0.3639, 0.0155, 0.9435, 0.5518]
    val_b = [0.2727, 0.0049, 0.1943, 0.0278, 0.9942, 0.5183]
    val_c = [0.9545, 0.8068, 0.9897, 0.6803, 0.9982, 0.9182]

    # Subplot 1 is polar
    ax1.remove()
    ax1 = fig.add_subplot(1, 2, 1, polar=True)

    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=9.5, fontweight="bold")

    # Plot lines
    ax1.plot(angles, val_a + val_a[:1], color=COLOR_A, linewidth=1.5, linestyle="--", label="Model A (1D CNN)")
    ax1.fill(angles, val_a + val_a[:1], color=COLOR_A, alpha=0.1)

    ax1.plot(angles, val_b + val_b[:1], color=COLOR_B, linewidth=1.5, linestyle="-.", label="Model B (CNN+GNN)")
    ax1.fill(angles, val_b + val_b[:1], color=COLOR_B, alpha=0.15)

    ax1.plot(angles, val_c + val_c[:1], color=COLOR_C, linewidth=2.5, linestyle="-", label="Model C (CNN+GNN+GRU)")
    ax1.fill(angles, val_c + val_c[:1], color=COLOR_C, alpha=0.25)

    ax1.set_ylim(0, 1.0)
    ax1.set_title("Multi-Metric Radar Comparison", fontweight="bold", pad=15)
    ax1.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1), fontsize=8.5)

    # Subplot 2: Waterfall / Component Gain
    deltas = [
        ("CNN Baseline F1", 0.0155, "#718096"),
        ("+ Spatial GNN", 0.0123, "#ED8936"),
        ("+ Causal GRU", 0.6525, "#2B6CB0"),
        ("Final Model C F1", 0.6803, "#2F855A")
    ]
    labels_wf = [d[0] for d in deltas]
    vals_wf = [d[1] for d in deltas]
    cols_wf = [d[2] for d in deltas]

    ax2.bar(labels_wf, vals_wf, color=cols_wf, edgecolor="black", width=0.55)
    ax2.set_ylabel("F1 Score Contribution", fontweight="bold")
    ax2.set_title("Architectural Component Value-Add (F1 Gain)", fontweight="bold")
    ax2.set_ylim(0, 0.8)
    for idx, (lbl, val, _) in enumerate(deltas):
        prefix = "+" if "+" in lbl else ""
        ax2.text(idx, val + 0.02, f"{prefix}{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    plt.suptitle("Figure 16: Master Architectural Ablation Synthesis", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig16_ablation_summary.png"))
    plt.close()
    print("  -> Generated fig16_ablation_summary.png")

    print("\nALL 16 PUBLICATION FIGURES SUCCESSFULLY GENERATED AT 300 DPI!")
    print("=" * 80)

if __name__ == "__main__":
    generate_all_figures()
