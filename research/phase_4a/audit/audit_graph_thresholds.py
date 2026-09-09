"""
NeuroAegis Phase 4A Audit: Graph Threshold & Topology Diagnostic Analysis
Tests candidate adjacency thresholds theta in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50].
Analyzes edge count, density, connected components, occipital lead disconnection, and degree distribution.
Generates:
  1. audit/graph_threshold_connectivity.png
  2. audit/graph_degree_distribution.png
  3. audit/graph_component_sizes.png
  4. audit/graph_threshold_sweep.json
  5. audit/training_correlation_matrix.npy
"""

import os
import sys
BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import mne
mne.set_log_level("ERROR")

from research.phase_2.chbmit_preprocessor import (
    CHBMITChannelManager,
    CHBMITSignalFilter,
    CHBMITNormalizer
)

AUDIT_DIR = os.path.join(BASE_DIR, "research/phase_4a/audit")
MANIFEST_DIR = os.path.join(BASE_DIR, "research/data/manifests")
CHANNEL_ORDER_PATH = os.path.join(BASE_DIR, "research/data/config/chbmit_channel_order.json")
SPLIT_CONFIG_PATH = os.path.join(BASE_DIR, "research/imbalance/class_imbalance_config.json")
EDF_ROOT = os.path.join(BASE_DIR, "CHB-MIT Dataset")
CORR_MATRIX_CACHE = os.path.join(AUDIT_DIR, "training_correlation_matrix.npy")


def get_training_correlation_matrix():
    if os.path.exists(CORR_MATRIX_CACHE):
        print(f"Loading cached training correlation matrix from {CORR_MATRIX_CACHE}...")
        return np.load(CORR_MATRIX_CACHE)
        
    print("Computing training correlation matrix across 16 training patients...")
    with open(CHANNEL_ORDER_PATH, "r") as f:
        channel_names = json.load(f)
    n_channels = len(channel_names)
    
    with open(SPLIT_CONFIG_PATH, "r") as f:
        split_cfg = json.load(f)
    train_patients = split_cfg["split_summary"]["train"]["patients"]
    
    manifest_df = pd.read_csv(os.path.join(MANIFEST_DIR, "chbmit_manifest.csv"))
    valid_train_recs = manifest_df[
        (manifest_df["patient_id"].isin(train_patients)) &
        (manifest_df["data_quality_status"] == "PASSED")
    ]
    
    channel_mgr = CHBMITChannelManager()
    signal_filter = CHBMITSignalFilter()
    normalizer = CHBMITNormalizer()
    
    corr_matrices = []
    for pat_id in train_patients:
        pat_recs = valid_train_recs[valid_train_recs["patient_id"] == pat_id]
        if len(pat_recs) == 0:
            continue
        for _, rec in pat_recs.iterrows():
            edf_path = os.path.join(EDF_ROOT, pat_id, rec["edf_filename"])
            if not os.path.exists(edf_path):
                continue
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                picks, status, missing = channel_mgr.map_recording_channels(raw.ch_names)
                if len(picks) == 23 and len(missing) == 0:
                    data = raw.get_data(picks=picks)
                    data = signal_filter.filter_signal(data)
                    data = normalizer.zscore_recording_local(data)
                    max_samples = min(data.shape[1], 256 * 1800)
                    sub_data = data[:, :max_samples]
                    corr = np.corrcoef(sub_data)
                    corr_matrices.append(corr)
                    print(f"  -> {pat_id}: loaded {rec['edf_filename']}")
                    break
            except Exception:
                continue
                
    avg_corr = np.mean(corr_matrices, axis=0)
    avg_corr = (avg_corr + avg_corr.T) / 2.0
    np.save(CORR_MATRIX_CACHE, avg_corr)
    print(f"Saved training correlation matrix to {CORR_MATRIX_CACHE}")
    return avg_corr


def analyze_thresholds():
    avg_corr = get_training_correlation_matrix()
    with open(CHANNEL_ORDER_PATH, "r") as f:
        channel_names = json.load(f)
    n_nodes = len(channel_names)
    
    abs_corr = np.abs(avg_corr)
    np.fill_diagonal(abs_corr, 0.0)
    
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    max_possible_edges = n_nodes * (n_nodes - 1) // 2
    
    sweep_results = []
    
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP ANALYSIS")
    print("=" * 80)
    print(f"{'Theta':<7} | {'Edges':<6} | {'Density':<8} | {'Comps':<6} | {'Occipital Connected?':<22} | {'Isolated Nodes'}")
    print("-" * 80)
    
    # Occipital node indices: P7-O1 (10), P3-O1 (14), P4-O2 (18), P8-O2 (21)
    occipital_names = ["P7-O1", "P3-O1", "P4-O2", "P8-O2"]
    occipital_indices = [channel_names.index(ch) for ch in occipital_names]
    
    threshold_graphs = {}
    
    for th in thresholds:
        A = np.where(abs_corr >= th, abs_corr, 0.0)
        
        # Build networkx graph
        G = nx.Graph()
        for i in range(n_nodes):
            G.add_node(i, label=channel_names[i])
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if A[i, j] > 0.0:
                    G.add_edge(i, j, weight=float(A[i, j]))
                    
        threshold_graphs[th] = G
        edges_count = G.number_of_edges()
        density = edges_count / max_possible_edges
        
        components = list(nx.connected_components(G))
        num_components = len(components)
        comp_sizes = sorted([len(c) for c in components], reverse=True)
        
        degrees = [G.degree(i) for i in range(n_nodes)]
        isolated = [channel_names[i] for i in range(n_nodes) if G.degree(i) == 0]
        
        # Check if occipital leads are in the main component (largest component)
        largest_comp = max(components, key=len)
        occipital_in_main = all(idx in largest_comp for idx in occipital_indices)
        
        status_str = "YES (Fully Connected)" if occipital_in_main else f"NO (Isolated/Sub-comp: {len([idx for idx in occipital_indices if idx in largest_comp])}/4 in main)"
        print(f"{th:<7.2f} | {edges_count:<6} | {density*100:<7.2f}% | {num_components:<6} | {status_str:<22} | {len(isolated)} nodes")
        
        comp_breakdowns = []
        for c_idx, c in enumerate(components, 1):
            c_channels = [channel_names[i] for i in sorted(c)]
            comp_breakdowns.append({"component_id": c_idx, "size": len(c), "channels": c_channels})
            
        sweep_results.append({
            "threshold": float(th),
            "edges": int(edges_count),
            "density": float(round(density, 4)),
            "num_components": int(num_components),
            "component_sizes": comp_sizes,
            "occipital_in_main_component": bool(occipital_in_main),
            "num_isolated_nodes": len(isolated),
            "isolated_channels": isolated,
            "mean_degree": float(round(np.mean(degrees), 2)),
            "min_degree": int(np.min(degrees)),
            "max_degree": int(np.max(degrees)),
            "components": comp_breakdowns
        })
        
    with open(os.path.join(AUDIT_DIR, "graph_threshold_sweep.json"), "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved threshold sweep JSON to {os.path.join(AUDIT_DIR, 'graph_threshold_sweep.json')}")
    
    # -------------------------------------------------------------
    # Figure 1: Connectivity vs Threshold Curve
    # -------------------------------------------------------------
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "axes.edgecolor": "#CBD5E1", "grid.color": "#F1F5F9"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    ths = [r["threshold"] for r in sweep_results]
    densities = [r["density"] * 100 for r in sweep_results]
    edges = [r["edges"] for r in sweep_results]
    n_comps = [r["num_components"] for r in sweep_results]
    max_comp_sizes = [r["component_sizes"][0] for r in sweep_results]
    
    # Left: Edges & Density
    color1 = "#2563EB"
    ax1.plot(ths, edges, marker="o", color=color1, linewidth=2.5, markersize=8, label="Undirected Edges")
    ax1.set_xlabel("Adjacency Threshold (θ)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Undirected Edges", color=color1, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax1_twin = ax1.twinx()
    color2 = "#10B981"
    ax1_twin.plot(ths, densities, marker="s", color=color2, linewidth=2.5, linestyle="--", markersize=8, label="Graph Density (%)")
    ax1_twin.set_ylabel("Graph Density (%)", color=color2, fontsize=11, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    
    # Highlight theta=0.35
    ax1.axvline(0.35, color="#EF4444", linestyle=":", linewidth=2, label="Phase 4A Frozen (θ=0.35)")
    ax1.set_title("Graph Density & Edge Count vs Threshold", fontsize=12, fontweight="bold", pad=12)
    
    # Right: Connected Components & Giant Component Size
    ax2.plot(ths, n_comps, marker="^", color="#DC2626", linewidth=2.5, markersize=8, label="Connected Components")
    ax2.plot(ths, max_comp_sizes, marker="v", color="#8B5CF6", linewidth=2.5, markersize=8, label="Giant Component Size (Nodes)")
    ax2.axvline(0.35, color="#EF4444", linestyle=":", linewidth=2, label="Phase 4A Frozen (θ=0.35)")
    ax2.set_xlabel("Adjacency Threshold (θ)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax2.set_title("Topological Fragmentation vs Threshold", fontsize=12, fontweight="bold", pad=12)
    ax2.legend(loc="center right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    # Annotate disconnection at theta=0.35
    ax2.annotate("θ=0.35: 2 Components\n(Occipital leads disconnected)",
                 xy=(0.35, 2), xytext=(0.37, 7),
                 arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9.5, fontweight="bold", color="#DC2626",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", edgecolor="#DC2626"))
                 
    plt.suptitle("Graph Threshold Diagnostic: Density, Connectivity & Fragmentation", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(AUDIT_DIR, "graph_threshold_connectivity.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1 to {fig1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Degree Distribution Comparison (θ=0.25 vs θ=0.30 vs θ=0.35)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    x = np.arange(n_nodes)
    width = 0.26
    
    g25 = threshold_graphs[0.25]
    g30 = threshold_graphs[0.30]
    g35 = threshold_graphs[0.35]
    
    deg25 = [g25.degree(i) for i in range(n_nodes)]
    deg30 = [g30.degree(i) for i in range(n_nodes)]
    deg35 = [g35.degree(i) for i in range(n_nodes)]
    
    ax.bar(x - width, deg25, width, label="Candidate θ=0.25 (Density 24.1%, 1 Component)", color="#3B82F6", edgecolor="#1E40AF")
    ax.bar(x, deg30, width, label="Candidate θ=0.30 (Density 17.8%, 1 Component)", color="#10B981", edgecolor="#065F46")
    ax.bar(x + width, deg35, width, label="Phase 4A θ=0.35 (Density 12.6%, 2 Components)", color="#EF4444", edgecolor="#991B1B")
    
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=9.5)
    ax.set_xlabel("EEG Channel (Bipolar Derivation)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Node Degree (# Connecting Edges)", fontsize=11, fontweight="bold")
    ax.set_title("Node Degree Distribution Across Candidate Graph Thresholds", fontsize=13, fontweight="bold", pad=15)
    ax.legend(frameon=True, fontsize=10.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    # Highlight occipital leads
    for occ_idx in occipital_indices:
        ax.get_xticklabels()[occ_idx].set_color("#DC2626")
        ax.get_xticklabels()[occ_idx].set_fontweight("bold")
        
    plt.tight_layout()
    fig2_path = os.path.join(AUDIT_DIR, "graph_degree_distribution.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2 to {fig2_path}")
    
    # -------------------------------------------------------------
    # Figure 3: Component Sizes Across Candidate Thresholds
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Plot stacked bar chart of component sizes
    for t_idx, th in enumerate(thresholds):
        comp_s = sorted([len(c) for c in nx.connected_components(threshold_graphs[th])], reverse=True)
        bottom = 0
        colors = ["#2563EB", "#EF4444", "#F59E0B", "#10B981", "#8B5CF6", "#64748B", "#EC4899", "#14B8A6"]
        for c_i, sz in enumerate(comp_s):
            col = colors[c_i % len(colors)]
            lbl = f"Component {c_i+1}" if t_idx == 0 else ""
            ax.bar(t_idx, sz, bottom=bottom, color=col, edgecolor="black", width=0.55, label=lbl if c_i < 3 and t_idx == 0 else "")
            if sz >= 2:
                ax.text(t_idx, bottom + sz / 2.0, f"{sz}", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
            bottom += sz
            
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"θ={th:.2f}" for th in thresholds], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Adjacency Threshold", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Number of Nodes (Total = 23)", fontsize=11, fontweight="bold")
    ax.set_title("Connected Component Partitioning Across Graph Thresholds", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 25)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    # Add annotation for phase 4a threshold
    ax.annotate("Phase 4A Frozen (θ=0.35)\nComponent 1: 19 nodes\nComponent 2: 4 nodes (Occipital)",
                xy=(4, 21), xytext=(4.3, 14),
                arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight="bold", color="#DC2626",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEE2E2", edgecolor="#DC2626"))
                
    plt.tight_layout()
    fig3_path = os.path.join(AUDIT_DIR, "graph_component_sizes.png")
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3 to {fig3_path}")
    print("=" * 80)
    print("THRESHOLD SWEEP COMPLETE!")

if __name__ == "__main__":
    analyze_thresholds()
