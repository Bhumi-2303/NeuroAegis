"""
NeuroAegis Phase 4A-C: Publication Figures Generator (300 DPI)
Generates 8 data-derived diagnostic figures in research/experiments/gnn_candidates/figures/:
  1. graph_topology_vs_threshold.png
  2. validation_auprc_vs_threshold.png
  3. validation_auroc_vs_threshold.png
  4. validation_event_sensitivity_vs_threshold.png
  5. validation_false_alarms_vs_threshold.png
  6. validation_confusion_matrix_theta025.png
  7. validation_confusion_matrix_theta030.png
  8. degree_distribution_threshold_comparison.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
PHASE4A_C_DIR = os.path.join(BASE_DIR, "research/experiments/gnn_candidates")
FIGURES_DIR = os.path.join(PHASE4A_C_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

GRAPH_CSV_PATH = os.path.join(PHASE4A_C_DIR, "graph_threshold_comparison.csv")
VAL_CSV_PATH = os.path.join(PHASE4A_C_DIR, "validation_threshold_comparison.csv")
CHANNEL_ORDER_PATH = os.path.join(BASE_DIR, "research/data/config/chbmit_channel_order.json")


def generate_all_figures():
    print("=" * 80)
    print("GENERATING PHASE 4A-C PUBLICATION FIGURES (300 DPI)")
    print("=" * 80)
    
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.0,
        "grid.color": "#F1F5F9",
        "grid.linestyle": "--",
        "grid.alpha": 0.7
    })
    
    df_graph = pd.read_csv(GRAPH_CSV_PATH)
    df_val = pd.read_csv(VAL_CSV_PATH)
    with open(CHANNEL_ORDER_PATH, "r") as f:
        channel_names = json.load(f)
        
    # Figure 1: graph_topology_vs_threshold.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    ths = df_graph["threshold"].values
    edges = df_graph["undirected_edges"].values
    densities = df_graph["graph_density_pct"].values
    comps = df_graph["num_connected_components"].values
    largest_comps = df_graph["largest_component_size"].values
    
    color1 = "#2563EB"
    ax1.plot(ths, edges, marker="o", color=color1, linewidth=2.5, markersize=8, label="Undirected Edges")
    ax1.set_xlabel("Adjacency Threshold (θ)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Undirected Edges", color=color1, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(ths)
    ax1.grid(True)
    
    ax1_twin = ax1.twinx()
    color2 = "#10B981"
    ax1_twin.plot(ths, densities, marker="s", color=color2, linewidth=2.5, linestyle="--", markersize=8, label="Graph Density (%)")
    ax1_twin.set_ylabel("Graph Density (%)", color=color2, fontsize=11, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("Edge Count & Density vs Graph Threshold", fontsize=12, fontweight="bold", pad=12)
    
    ax2.bar(ths - 0.008, comps, width=0.015, color="#EF4444", edgecolor="#991B1B", label="Connected Components")
    ax2.set_xlabel("Adjacency Threshold (θ)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Component Count", color="#EF4444", fontsize=11, fontweight="bold")
    ax2.set_xticks(ths)
    ax2.set_ylim(0, 4)
    ax2.grid(True)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(ths, largest_comps, marker="^", color="#8B5CF6", linewidth=2.5, markersize=8, label="Giant Component Size")
    ax2_twin.set_ylabel("Giant Component Size (Nodes)", color="#8B5CF6", fontsize=11, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor="#8B5CF6")
    ax2_twin.set_ylim(15, 24)
    ax2.set_title("Topological Fragmentation vs Threshold", fontsize=12, fontweight="bold", pad=12)
    
    plt.suptitle("Figure 1: Graph Topology & Fragmentation vs Candidate Thresholds", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "graph_topology_vs_threshold.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1: {fig1_path}")
    
    # Figure 2: validation_auprc_vs_threshold.png
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    c_labels = [f"θ={th:.2f}" for th in df_val["threshold"]]
    auprcs = df_val["validation_auprc"].values
    bar_colors = ["#3B82F6", "#10B981", "#64748B"]
    bars = ax.bar(c_labels, auprcs, color=bar_colors, edgecolor="#1E293B", width=0.5)
    ax.set_ylabel("Validation AUPRC (Primary Metric)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 2: Validation AUPRC Across Graph Thresholds", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, axis="y")
    for bar, val in zip(bars, auprcs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(auprcs)*0.03, f"{val:.5f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(auprcs) * 1.25)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "validation_auprc_vs_threshold.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2: {fig2_path}")
    
    # Figure 3: validation_auroc_vs_threshold.png
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    aurocs = df_val["validation_auroc"].values
    bars = ax.bar(c_labels, aurocs, color=bar_colors, edgecolor="#1E293B", width=0.5)
    ax.set_ylabel("Validation AUROC", fontsize=11, fontweight="bold")
    ax.set_title("Figure 3: Validation AUROC Across Graph Thresholds", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, axis="y")
    for bar, val in zip(bars, aurocs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(aurocs)*0.03, f"{val:.5f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(aurocs) * 1.25)
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "validation_auroc_vs_threshold.png")
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3: {fig3_path}")
    
    # Figure 4: validation_event_sensitivity_vs_threshold.png
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    event_sens = df_val["validation_event_sensitivity"].values * 100
    bars = ax.bar(c_labels, event_sens, color=["#10B981", "#3B82F6", "#64748B"], edgecolor="#1E293B", width=0.5)
    ax.set_ylabel("Validation Event Sensitivity (%)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 4: Seizure Event Sensitivity on Validation Set (25 Events)", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, axis="y")
    ax.set_ylim(0, 115)
    for bar, val in zip(bars, event_sens):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "validation_event_sensitivity_vs_threshold.png")
    plt.savefig(fig4_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4: {fig4_path}")
    
    # Figure 5: validation_false_alarms_vs_threshold.png
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    fa_rates = df_val["validation_false_alarms_per_day"].values
    bars = ax.bar(c_labels, fa_rates, color=["#F59E0B", "#F59E0B", "#64748B"], edgecolor="#1E293B", width=0.5)
    ax.set_ylabel("Validation False Alarms / 24h", fontsize=11, fontweight="bold")
    ax.set_title("Figure 5: Validation False Alarm Rate Across Thresholds", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, axis="y")
    for bar, val in zip(bars, fa_rates):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(fa_rates)*0.03, f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(fa_rates) * 1.25)
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "validation_false_alarms_vs_threshold.png")
    plt.savefig(fig5_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 5: {fig5_path}")
    
    # Figure 6: validation_confusion_matrix_theta025.png
    row_025 = df_val[df_val["threshold"] == 0.25].iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    cm_025 = np.array([[row_025["tn"], row_025["fp"]], [row_025["fn"], row_025["tp"]]])
    im = ax.imshow(cm_025, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Background (0)", "Pred Seizure (1)"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["True Background (0)", "True Seizure (1)"], fontsize=10, fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = cm_025[i, j]
            color = "white" if val > cm_025.max()/2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontweight="bold", fontsize=11)
    ax.set_title(f"Figure 6: Validation Confusion Matrix (θ=0.25, 293,410 Windows)\nSens={row_025['validation_sensitivity']*100:.2f}%, Spec={row_025['validation_specificity']*100:.2f}%", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig6_path = os.path.join(FIGURES_DIR, "validation_confusion_matrix_theta025.png")
    plt.savefig(fig6_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 6: {fig6_path}")
    
    # Figure 7: validation_confusion_matrix_theta030.png
    row_030 = df_val[df_val["threshold"] == 0.30].iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    cm_030 = np.array([[row_030["tn"], row_030["fp"]], [row_030["fn"], row_030["tp"]]])
    im = ax.imshow(cm_030, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Background (0)", "Pred Seizure (1)"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["True Background (0)", "True Seizure (1)"], fontsize=10, fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = cm_030[i, j]
            color = "white" if val > cm_030.max()/2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontweight="bold", fontsize=11)
    ax.set_title(f"Figure 7: Validation Confusion Matrix (θ=0.30, 293,410 Windows)\nSens={row_030['validation_sensitivity']*100:.2f}%, Spec={row_030['validation_specificity']*100:.2f}%", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig7_path = os.path.join(FIGURES_DIR, "validation_confusion_matrix_theta030.png")
    plt.savefig(fig7_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 7: {fig7_path}")
    
    # Figure 8: degree_distribution_threshold_comparison.png
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    x = np.arange(len(channel_names))
    width = 0.26
    
    # Load adjacency matrices to compute node degrees
    adj25 = pd.read_csv(os.path.join(PHASE4A_C_DIR, "theta_025/graph_adjacency.csv"), index_col=0).values
    adj30 = pd.read_csv(os.path.join(PHASE4A_C_DIR, "theta_030/graph_adjacency.csv"), index_col=0).values
    adj35 = pd.read_csv(os.path.join(BASE_DIR, "research/experiments/gnn/cnn_gnn/exp_01/graph_adjacency.csv"), index_col=0).values
    
    deg25 = [int(np.sum(adj25[i, :] > 0) - 1) for i in range(23)]
    deg30 = [int(np.sum(adj30[i, :] > 0) - 1) for i in range(23)]
    deg35 = [int(np.sum(adj35[i, :] > 0) - 1) for i in range(23)]
    
    ax.bar(x - width, deg25, width, label="Candidate θ=0.25 (Density 23.7%, 60 edges)", color="#3B82F6", edgecolor="#1E40AF")
    ax.bar(x, deg30, width, label="Candidate θ=0.30 (Density 15.8%, 40 edges)", color="#10B981", edgecolor="#065F46")
    ax.bar(x + width, deg35, width, label="Frozen Ref θ=0.35 (Density 12.6%, 32 edges)", color="#EF4444", edgecolor="#991B1B")
    
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=9.5)
    ax.set_xlabel("EEG Channel (Bipolar Derivation)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Node Degree (# Connecting Edges)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 8: Node Degree Distribution Across Candidate Graph Thresholds", fontsize=13, fontweight="bold", pad=15)
    ax.legend(frameon=True, fontsize=10.5)
    ax.grid(True, axis="y")
    
    plt.tight_layout()
    fig8_path = os.path.join(FIGURES_DIR, "degree_distribution_threshold_comparison.png")
    plt.savefig(fig8_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 8: {fig8_path}")
    print("=" * 80)
    print("ALL 8 FIGURES GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    generate_all_figures()
