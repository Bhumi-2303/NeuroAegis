"""
NeuroAegis Phase 4A-C: Final Publication Figures Generator (300 DPI)
Generates 13 publication figures in research/experiments/gnn/figures/:
  1. graph_topology_vs_threshold.png
  2. degree_distribution_threshold_comparison.png
  3. graph_topology_visualization_candidates.png
  4. validation_auprc_vs_threshold.png
  5. validation_auroc_vs_threshold.png
  6. validation_event_sensitivity_vs_threshold.png
  7. validation_false_alarms_vs_threshold.png
  8. validation_confusion_matrices.png
  9. final_selected_graph_topology.png
  10. final_test_confusion_matrix_raw.png
  11. final_test_confusion_matrix_normalized.png
  12. final_test_roc.png
  13. final_test_pr.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
FIGURES_DIR = os.path.join(BASE_DIR, "research/experiments/gnn/figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

GRAPH_CSV_PATH = os.path.join(BASE_DIR, "research/experiments/gnn_candidates/graph_threshold_comparison.csv")
VAL_CSV_PATH = os.path.join(BASE_DIR, "research/experiments/gnn_candidates/validation_threshold_comparison.csv")
TEST_METRICS_PATH = os.path.join(BASE_DIR, "research/experiments/gnn/final_test_metrics.json")
TEST_PREDS_PATH = os.path.join(BASE_DIR, "research/experiments/gnn/final_test_predictions.npz")
CHANNEL_ORDER_PATH = os.path.join(BASE_DIR, "research/data/config/chbmit_channel_order.json")

def generate_all_figures():
    print("=" * 80)
    print("GENERATING PHASE 4A-C FINAL PUBLICATION FIGURES (300 DPI)")
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
    with open(CHANNEL_ORDER_PATH) as f:
        channel_names = json.load(f)
    with open(TEST_METRICS_PATH) as f:
        test_metrics = json.load(f)
    test_npz = np.load(TEST_PREDS_PATH)
    test_true = test_npz["y_true"]
    test_prob = test_npz["y_prob"]
    
    # -------------------------------------------------------------
    # FIGURE 1: Graph Topology and Fragmentation vs Threshold
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # FIGURE 2: Node Degree Distribution Across Candidate Thresholds
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    x = np.arange(len(channel_names))
    width = 0.26
    
    adj25 = pd.read_csv(os.path.join(BASE_DIR, "research/experiments/gnn_candidates/theta_025/graph_adjacency.csv"), index_col=0).values
    adj30 = pd.read_csv(os.path.join(BASE_DIR, "research/experiments/gnn_candidates/theta_030/graph_adjacency.csv"), index_col=0).values
    adj35 = pd.read_csv(os.path.join(BASE_DIR, "research/experiments/gnn/cnn_gnn/exp_01/graph_adjacency.csv"), index_col=0).values
    
    deg25 = [int(np.sum(adj25[i, :] > 0) - 1) for i in range(23)]
    deg30 = [int(np.sum(adj30[i, :] > 0) - 1) for i in range(23)]
    deg35 = [int(np.sum(adj35[i, :] > 0) - 1) for i in range(23)]
    
    ax.bar(x - width, deg25, width, label="θ = 0.25 (Density 23.72%, 60 edges)", color="#3B82F6", edgecolor="#1E40AF")
    ax.bar(x, deg30, width, label="θ = 0.30 [Selected] (Density 15.81%, 40 edges)", color="#10B981", edgecolor="#065F46")
    ax.bar(x + width, deg35, width, label="θ = 0.35 [Ref] (Density 12.65%, 32 edges)", color="#EF4444", edgecolor="#991B1B")
    
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=9.5)
    ax.set_xlabel("EEG Bipolar Channel Derivation", fontsize=11, fontweight="bold")
    ax.set_ylabel("Node Degree (# Connecting Edges)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 2: Node Degree Distribution Across Candidate Graph Thresholds", fontsize=13, fontweight="bold", pad=15)
    ax.legend(frameon=True, fontsize=10.5)
    ax.grid(True, axis="y")
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "degree_distribution_threshold_comparison.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2: {fig2_path}")

    # -------------------------------------------------------------
    # FIGURE 3: Graph Topology Visualizations for Each Candidate
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=300)
    candidates_info = [
        ("θ = 0.25 (60 edges)", adj25, "#3B82F6"),
        ("θ = 0.30 [Selected] (40 edges)", adj30, "#10B981"),
        ("θ = 0.35 [Reference] (32 edges)", adj35, "#EF4444")
    ]
    
    # Compute deterministic circular layout
    G_template = nx.Graph()
    for i in range(23):
        G_template.add_node(i)
    pos = nx.circular_layout(G_template)
    
    for ax, (title, mat, edge_col) in zip(axes, candidates_info):
        G = nx.Graph()
        for i in range(23):
            G.add_node(i, label=channel_names[i])
        for i in range(23):
            for j in range(i+1, 23):
                if mat[i, j] > 0:
                    G.add_edge(i, j, weight=float(mat[i, j]))
                    
        # Node colors: Component 1 (blue), Component 2 (red)
        comps = list(nx.connected_components(G))
        node_colors = []
        for i in range(23):
            if i in comps[0]:
                node_colors.append("#DBEAFE" if len(comps[0]) > len(comps[1]) else "#FEE2E2")
            else:
                node_colors.append("#FEE2E2" if len(comps[0]) > len(comps[1]) else "#DBEAFE")
                
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=800, edgecolors="#1E293B", linewidths=1.5)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_col, width=1.8, alpha=0.7)
        
        # Non-overlapping labels with radial displacement
        labels = {i: channel_names[i] for i in range(23)}
        for i, (lx, ly) in pos.items():
            # offset label slightly outward
            r = np.sqrt(lx**2 + ly**2)
            if r > 0:
                off_x = lx * 1.18
                off_y = ly * 1.18
            else:
                off_x, off_y = lx, ly
            ax.text(off_x, off_y, channel_names[i], fontsize=8, fontweight="bold",
                    ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CBD5E1", alpha=0.9))
                    
        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
        ax.axis("off")
        
    plt.suptitle("Figure 3: Spatial Graph Topologies Across Candidate Adjacency Thresholds", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "graph_topology_visualization_candidates.png")
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3: {fig3_path}")

    # -------------------------------------------------------------
    # FIGURE 4: Validation AUPRC Across Candidate Thresholds
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    c_labels = [f"θ = {th:.2f}" for th in df_val["threshold"]]
    auprcs = df_val["validation_auprc"].values
    bars = ax.bar(c_labels, auprcs, color=["#3B82F6", "#10B981", "#64748B"], edgecolor="#1E293B", width=0.45)
    ax.set_ylabel("Validation AUPRC (Primary Metric)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 4: Validation AUPRC Across Candidate Thresholds (293,410 Windows)", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, axis="y")
    for bar, val in zip(bars, auprcs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(auprcs)*0.03, f"{val:.5f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max(auprcs) * 1.25)
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "validation_auprc_vs_threshold.png")
    plt.savefig(fig4_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4: {fig4_path}")

    # -------------------------------------------------------------
    # FIGURE 5: Validation AUROC Across Candidate Thresholds
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    aurocs = df_val["validation_auroc"].values
    bars = ax.bar(c_labels, aurocs, color=["#3B82F6", "#10B981", "#64748B"], edgecolor="#1E293B", width=0.45)
    ax.set_ylabel("Validation AUROC", fontsize=11, fontweight="bold")
    ax.set_title("Figure 5: Validation AUROC Across Candidate Thresholds", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, axis="y")
    for bar, val in zip(bars, aurocs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(aurocs)*0.03, f"{val:.5f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max(aurocs) * 1.25)
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "validation_auroc_vs_threshold.png")
    plt.savefig(fig5_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 5: {fig5_path}")

    # -------------------------------------------------------------
    # FIGURE 6: Validation Event Sensitivity Across Candidate Thresholds
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ev_sens = df_val["validation_event_sensitivity"].values * 100
    bars = ax.bar(c_labels, ev_sens, color=["#3B82F6", "#10B981", "#64748B"], edgecolor="#1E293B", width=0.45)
    ax.set_ylabel("Validation Event Sensitivity (%)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 6: Seizure Event Sensitivity on Validation Set (25 Events)", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, axis="y")
    ax.set_ylim(0, 60)
    for bar, val in zip(bars, ev_sens):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    fig6_path = os.path.join(FIGURES_DIR, "validation_event_sensitivity_vs_threshold.png")
    plt.savefig(fig6_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 6: {fig6_path}")

    # -------------------------------------------------------------
    # FIGURE 7: Validation False Alarms per 24 Hours
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    fa_rates = df_val["validation_false_alarms_per_day"].values
    bars = ax.bar(c_labels, fa_rates, color=["#F59E0B", "#F59E0B", "#64748B"], edgecolor="#1E293B", width=0.45)
    ax.set_ylabel("False Alarms / 24 Hours", fontsize=11, fontweight="bold")
    ax.set_title("Figure 7: Validation False Alarm Rate Across Candidate Thresholds", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, axis="y")
    for bar, val in zip(bars, fa_rates):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(fa_rates)*0.03, f"{val:.1f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max(fa_rates) * 1.25)
    plt.tight_layout()
    fig7_path = os.path.join(FIGURES_DIR, "validation_false_alarms_vs_threshold.png")
    plt.savefig(fig7_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 7: {fig7_path}")

    # -------------------------------------------------------------
    # FIGURE 8: Validation Confusion Matrices
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    for ax, (_, row) in zip(axes, df_val.iterrows()):
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Background (0)", "Pred Seizure (1)"], fontsize=9.5, fontweight="bold")
        ax.set_yticklabels(["True Background (0)", "True Seizure (1)"], fontsize=9.5, fontweight="bold")
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                clr = "white" if val > cm.max()/2 else "black"
                ax.text(j, i, f"{val:,}", ha="center", va="center", color=clr, fontweight="bold", fontsize=10.5)
        ax.set_title(f"θ = {row['threshold']:.2f} (Val)\nSens={row['validation_sensitivity']*100:.2f}%, Spec={row['validation_specificity']*100:.2f}%", fontsize=11, fontweight="bold", pad=10)
    plt.suptitle("Figure 8: Validation Confusion Matrices Across Candidate Thresholds (293,410 Windows)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig8_path = os.path.join(FIGURES_DIR, "validation_confusion_matrices.png")
    plt.savefig(fig8_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 8: {fig8_path}")

    # -------------------------------------------------------------
    # FIGURE 9: Final Selected Graph Topology (θ = 0.30)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    G_sel = nx.Graph()
    for i in range(23):
        G_sel.add_node(i, label=channel_names[i])
    for i in range(23):
        for j in range(i+1, 23):
            if adj30[i, j] > 0:
                G_sel.add_edge(i, j, weight=float(adj30[i, j]))
                
    comps = list(nx.connected_components(G_sel))
    node_colors = ["#DBEAFE" if i in comps[0] else "#FEE2E2" for i in range(23)]
    
    nx.draw_networkx_nodes(G_sel, pos, ax=ax, node_color=node_colors, node_size=1100, edgecolors="#1E293B", linewidths=2)
    nx.draw_networkx_edges(G_sel, pos, ax=ax, edge_color="#10B981", width=2.2, alpha=0.8)
    
    for i, (lx, ly) in pos.items():
        r = np.sqrt(lx**2 + ly**2)
        off_x = lx * 1.20 if r > 0 else lx
        off_y = ly * 1.20 if r > 0 else ly
        deg_i = G_sel.degree(i)
        ax.text(off_x, off_y, f"{channel_names[i]}\n(deg {deg_i})", fontsize=8, fontweight="bold",
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#10B981", alpha=0.95))
                
    ax.set_title("Figure 9: Final Selected & Frozen Spatial Graph Topology (θ = 0.30)\n40 Undirected Edges, Density 15.81%, 2 Components (19 Nodes Anterior, 4 Nodes Occipital)",
                 fontsize=11.5, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    fig9_path = os.path.join(FIGURES_DIR, "final_selected_graph_topology.png")
    plt.savefig(fig9_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 9: {fig9_path}")

    # -------------------------------------------------------------
    # FIGURE 10 & 11: Final Test Confusion Matrices (Raw & Normalized)
    # -------------------------------------------------------------
    y_pred_test = (test_prob >= 0.50).astype(int)
    cm_test_raw = confusion_matrix(test_true, y_pred_test, labels=[0, 1])
    cm_test_norm = cm_test_raw.astype(float) / cm_test_raw.sum(axis=1, keepdims=True)
    
    # Raw
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(cm_test_raw, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Background (0)", "Pred Seizure (1)"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["True Background (0)", "True Seizure (1)"], fontsize=10, fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = cm_test_raw[i, j]
            clr = "white" if val > cm_test_raw.max()/2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=clr, fontweight="bold", fontsize=11)
    ax.set_title(f"Figure 10: Final Test Raw Confusion Matrix (Frozen θ = 0.30)\nTotal Windows: 219,909 | TP={test_metrics['confusion_matrix']['tp']:,}, FP={test_metrics['confusion_matrix']['fp']:,}", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig10_path = os.path.join(FIGURES_DIR, "final_test_confusion_matrix_raw.png")
    plt.savefig(fig10_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 10: {fig10_path}")
    
    # Normalized
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(cm_test_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Background (0)", "Pred Seizure (1)"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["True Background (0)", "True Seizure (1)"], fontsize=10, fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = cm_test_norm[i, j]
            clr = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val*100:.2f}%", ha="center", va="center", color=clr, fontweight="bold", fontsize=11)
    ax.set_title(f"Figure 11: Final Test Normalized Confusion Matrix (Frozen θ = 0.30)\nSensitivity: {test_metrics['test_sensitivity']*100:.2f}% | Specificity: {test_metrics['test_specificity']*100:.2f}%", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig11_path = os.path.join(FIGURES_DIR, "final_test_confusion_matrix_normalized.png")
    plt.savefig(fig11_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 11: {fig11_path}")

    # -------------------------------------------------------------
    # FIGURE 12: Final Test ROC Curve
    # -------------------------------------------------------------
    fpr, tpr, _ = roc_curve(test_true, test_prob)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(fpr, tpr, color="#2563EB", linewidth=2.5, label=f"Frozen CNN+GNN θ=0.30 (AUROC = {test_metrics['test_auroc']:.5f})")
    ax.plot([0, 1], [0, 1], color="#DC2626", linestyle="--", linewidth=1.5, label="Random Classifier (AUROC = 0.5000)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 12: Final Test Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, fontsize=10.5)
    ax.grid(True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig12_path = os.path.join(FIGURES_DIR, "final_test_roc.png")
    plt.savefig(fig12_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 12: {fig12_path}")

    # -------------------------------------------------------------
    # FIGURE 13: Final Test Precision-Recall Curve
    # -------------------------------------------------------------
    prevalence = np.mean(test_true)
    prec_pts, rec_pts, _ = precision_recall_curve(test_true, test_prob)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(rec_pts, prec_pts, color="#10B981", linewidth=2.5, label=f"Frozen CNN+GNN θ=0.30 (AUPRC = {test_metrics['test_auprc']:.5f})")
    ax.axhline(prevalence, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Positive Prevalence Baseline ({prevalence*100:.3f}%)")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("Figure 13: Final Test Precision-Recall (PR) Curve", fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="upper right", frameon=True, fontsize=10.5)
    ax.grid(True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, max(max(prec_pts)*1.1, 0.05)])
    plt.tight_layout()
    fig13_path = os.path.join(FIGURES_DIR, "final_test_pr.png")
    plt.savefig(fig13_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 13: {fig13_path}")
    print("=" * 80)
    print("ALL 13 PUBLICATION FIGURES SUCCESSFULLY GENERATED!")

if __name__ == "__main__":
    generate_all_figures()
