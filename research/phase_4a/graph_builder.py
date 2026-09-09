"""
NeuroAegis Phase 4A: Static EEG Spatial Graph Builder
Computes cross-channel Pearson correlation strictly on CHB-MIT training patients,
constructs thresholded and normalized adjacency matrices with self-loops,
and exports graph_config.json, graph_adjacency.csv, and publication figures.
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

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
EXP_DIR = os.path.join(BASE_DIR, "research/phase_4a/cnn_gnn/exp_01")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")
MANIFEST_DIR = os.path.join(BASE_DIR, "research/data/manifests")
CHANNEL_ORDER_PATH = os.path.join(BASE_DIR, "research/data/config/chbmit_channel_order.json")
SPLIT_CONFIG_PATH = os.path.join(BASE_DIR, "research/imbalance/class_imbalance_config.json")
EDF_ROOT = os.path.join(BASE_DIR, "CHB-MIT Dataset")


def build_spatial_graph(
    threshold: float = 0.35,
    save_dir: str = EXP_DIR,
    figures_dir: str = FIGURES_DIR
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    with open(CHANNEL_ORDER_PATH, "r") as f:
        channel_names = json.load(f)
    n_channels = len(channel_names)
    assert n_channels == 23, f"Expected 23 channels, found {n_channels}"
    
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
    print(f"Estimating Pearson correlation across {len(train_patients)} training patients...")
    
    for pat_id in train_patients:
        pat_recs = valid_train_recs[valid_train_recs["patient_id"] == pat_id]
        if len(pat_recs) == 0:
            continue
            
        found_canonical = False
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
                    print(f"  -> Processed {pat_id} ({rec['edf_filename']}): status {status}, matrix shape {corr.shape}")
                    found_canonical = True
                    break
            except Exception as e:
                continue
                
        if not found_canonical:
            print(f"  -> Warning: Could not find canonical 23-channel recording for {pat_id}")
        
    avg_corr = np.mean(corr_matrices, axis=0)
    # Enforce exact symmetry and zero diagonal for edge matrix
    avg_corr = (avg_corr + avg_corr.T) / 2.0
    abs_corr = np.abs(avg_corr)
    np.fill_diagonal(abs_corr, 0.0)
    
    # Construct binary / thresholded adjacency
    A = np.where(abs_corr >= threshold, abs_corr, 0.0)
    
    # Ensure no isolated nodes: connect any isolated node to its highest correlation neighbor
    for i in range(n_channels):
        if np.sum(A[i, :]) == 0.0:
            best_j = int(np.argmax(abs_corr[i, :]))
            A[i, best_j] = abs_corr[i, best_j]
            A[best_j, i] = abs_corr[i, best_j]
            print(f"  -> Connected isolated node {channel_names[i]} to {channel_names[best_j]} (weight {abs_corr[i, best_j]:.4f})")
            
    # Self-loops: A_tilde = A + I
    A_tilde = A + np.eye(n_channels, dtype=np.float32)
    degrees = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Normalized adjacency: D^{-1/2} A_tilde D^{-1/2}
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
    # Compute graph statistics
    binary_edges = int(np.sum(A > 0) / 2)
    max_possible_edges = n_channels * (n_channels - 1) // 2
    density = float(binary_edges / max_possible_edges)
    node_degrees = [int(np.sum(A[i, :] > 0)) for i in range(n_channels)]
    
    # Save graph_adjacency.csv
    adj_csv_path = os.path.join(save_dir, "graph_adjacency.csv")
    df_adj = pd.DataFrame(A_norm, index=channel_names, columns=channel_names)
    df_adj.to_csv(adj_csv_path)
    print(f"Saved normalized adjacency matrix to {adj_csv_path}")
    
    # Save graph_config.json
    graph_config = {
        "dataset": "CHB-MIT",
        "num_nodes": n_channels,
        "channel_order": channel_names,
        "adjacency_threshold": float(threshold),
        "total_undirected_edges": binary_edges,
        "max_possible_edges": max_possible_edges,
        "graph_density": float(round(density, 4)),
        "mean_degree": float(round(np.mean(node_degrees), 2)),
        "min_degree": int(np.min(node_degrees)),
        "max_degree": int(np.max(node_degrees)),
        "node_degrees_by_index": [{"index": i, "channel": channel_names[i], "degree": node_degrees[i]} for i in range(n_channels)],
        "node_degrees": {f"{channel_names[i]} (ch{i+1})": node_degrees[i] for i in range(n_channels)},
        "normalization": "Symmetric Kipf-Welling (D^-0.5 * (A + I) * D^-0.5)",
        "training_patients_used": train_patients,
        "label_leakage": "NONE (computed strictly on unlabelled EEG data of training set)"
    }
    config_json_path = os.path.join(save_dir, "graph_config.json")
    with open(config_json_path, "w") as f:
        json.dump(graph_config, f, indent=2)
    print(f"Saved graph config to {config_json_path}")
    
    # Figure 1: Graph Adjacency Heatmap
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    cax = ax.imshow(A_norm, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(n_channels))
    ax.set_yticks(range(n_channels))
    ax.set_xticklabels(channel_names, rotation=90, fontsize=8, fontweight="bold")
    ax.set_yticklabels(channel_names, fontsize=8, fontweight="bold")
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Normalized Adjacency Weight $\\hat{A}_{ij}$", fontsize=9, fontweight="bold")
    ax.set_title(f"Figure 1: Normalized Spatial Adjacency Matrix (\\theta={threshold}, Density={density:.2f})", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "fig01_graph_adjacency_heatmap.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 1: {fig1_path}")
    
    # Figure 2: EEG Channel Graph Topology
    G = nx.Graph()
    node_ids = [f"{i:02d}_{ch}" for i, ch in enumerate(channel_names)]
    node_labels = {node_ids[i]: (ch if i != 22 else "T8-P8 (2)") for i, ch in enumerate(channel_names)}
    for nid in node_ids:
        G.add_node(nid)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if A[i, j] > 0:
                G.add_edge(node_ids[i], node_ids[j], weight=float(A[i, j]))
                
    fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
    pos = nx.spring_layout(G, seed=42, k=0.55, iterations=100)
    node_colors = ["#3B82F6" if deg > 2 else "#10B981" for deg in node_degrees]
    edge_weights = [d["weight"] * 3.5 for (u, v, d) in G.edges(data=True)]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=750, alpha=0.9, edgecolors="#1E293B", linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color="#64748B", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7.5, font_family="sans-serif", font_weight="bold", font_color="#FFFFFF", ax=ax)
    
    ax.set_title(f"Figure 2: EEG 23-Channel Spatial Graph Topology ({binary_edges} Edges, Density: {density*100:.1f}%)", fontsize=12, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "fig02_eeg_channel_graph.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 2: {fig2_path}")
    
    return {
        "A_norm": A_norm,
        "config": graph_config,
        "adj_csv_path": adj_csv_path,
        "config_json_path": config_json_path
    }


if __name__ == "__main__":
    build_spatial_graph()
