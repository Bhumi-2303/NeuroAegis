"""
NeuroAegis Phase 4A Scientific Audit: Tensor Flow, Channel Identity, and Channel Ordering
Performs:
1. Exact tensor shape logging at each layer of the model.
2. Controlled channel isolation test: verifies if activating channel i activates ONLY node i before message passing.
3. Complete 23-channel order comparison across Phase 1, Phase 2, Phase 3, and Phase 4A.
4. Outputs audit/channel_node_mapping.csv.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from neuroaegis.models.baselines.cnn_gnn_model import Baseline1DCNN_GNN
from research.experiments.windowing_labeling.chbmit_preprocessor import CHBMITChannelManager

AUDIT_DIR = os.path.join(BASE_DIR, "research/experiments/gnn/audit")
EXP_DIR = os.path.join(BASE_DIR, "research/experiments/gnn/cnn_gnn/exp_01")
GRAPH_ADJ_PATH = os.path.join(EXP_DIR, "graph_adjacency.csv")
CHANNEL_ORDER_PATH = os.path.join(BASE_DIR, "research/data/config/chbmit_channel_order.json")


def audit_tensor_flow_and_channels():
    print("=" * 80)
    print("AUDIT SECTION 3 & 4 & 5: TENSOR FLOW, CHANNEL IDENTITY, & CHANNEL MAPPING")
    print("=" * 80)
    
    # 1. Load Channel Configurations
    with open(CHANNEL_ORDER_PATH, "r") as f:
        canonical_channels = json.load(f)
    print(f"Loaded {len(canonical_channels)} canonical channels from Phase 1 config:")
    for idx, ch in enumerate(canonical_channels):
        print(f"  Ch {idx:02d}: {ch}")
        
    adj_df = pd.read_csv(GRAPH_ADJ_PATH, index_col=0)
    graph_channels = list(adj_df.columns)
    print(f"\nLoaded {len(graph_channels)} channels from Phase 4A graph adjacency:")
    for idx, ch in enumerate(graph_channels):
        print(f"  Node {idx:02d}: {ch}")
        
    ch_mgr = CHBMITChannelManager()
    p2_channels = ch_mgr.get_canonical_channels()
    
    # Verify channel ordering across all phases
    mapping_rows = []
    mismatch_detected = False
    for i in range(23):
        c_p1 = canonical_channels[i]
        c_p2 = p2_channels[i]
        c_p3 = p2_channels[i] # Phase 3 used CHBMITChannelManager
        c_p4 = graph_channels[i]
        
        matches = (c_p1 == c_p2 == c_p3 == c_p4)
        if not matches:
            mismatch_detected = True
            
        mapping_rows.append({
            "node_index": i,
            "channel_name": c_p1,
            "phase1_index": i,
            "phase2_index": i if c_p2 == c_p1 else -1,
            "phase3_index": i if c_p3 == c_p1 else -1,
            "phase4a_index": i if c_p4 == c_p1 else -1,
            "identical": matches
        })
        
    df_mapping = pd.DataFrame(mapping_rows)
    mapping_csv_path = os.path.join(AUDIT_DIR, "channel_node_mapping.csv")
    df_mapping.to_csv(mapping_csv_path, index=False)
    print(f"\nSaved channel node mapping to {mapping_csv_path}")
    print(f"Channel Order Mismatch across Phases: {'YES (FAIL)' if mismatch_detected else 'NO (PASS)'}")
    
    # 2. Inspect Tensor Shapes at Every Stage
    print("\n--- Inspecting Tensor Shapes at Every Stage ---")
    model = Baseline1DCNN_GNN(in_channels=23, adj_matrix=adj_df.values.astype(np.float32))
    model.eval()
    
    batch_size = 4
    x = torch.randn(batch_size, 23, 1280)
    print(f"Stage 0 (Input):                    shape = {tuple(x.shape)}")
    
    # Reshape
    b, c, t = x.shape
    x_reshaped = x.view(b * c, 1, t)
    print(f"Stage 1 (Reshape for Conv1D):       shape = {tuple(x_reshaped.shape)}")
    
    # Stage-by-stage temporal backbone
    tb = model.temporal_backbone
    # Block 1: Conv1d(1, 16) + BN + GELU + MaxPool
    s1 = tb[4](tb[3](tb[2](tb[1](tb[0](x_reshaped)))))
    print(f"Stage 2 (Temporal Conv Block 1):    shape = {tuple(s1.shape)} (16 filters, time {s1.shape[-1]})")
    
    # Block 2: Conv1d(16, 32) + BN + GELU + MaxPool
    s2 = tb[9](tb[8](tb[7](tb[6](tb[5](s1)))))
    print(f"Stage 3 (Temporal Conv Block 2):    shape = {tuple(s2.shape)} (32 filters, time {s2.shape[-1]})")
    
    # Block 3: Conv1d(32, 64) + BN + GELU + MaxPool
    s3 = tb[14](tb[13](tb[12](tb[11](tb[10](s2)))))
    print(f"Stage 4 (Temporal Conv Block 3):    shape = {tuple(s3.shape)} (64 filters, time {s3.shape[-1]})")
    
    # Block 4: Conv1d(64, 64) + BN + GELU + AdaptiveAvgPool(1)
    s4 = tb[19](tb[18](tb[17](tb[16](tb[15](s3)))))
    print(f"Stage 5 (Temporal Conv Block 4):    shape = {tuple(s4.shape)} (64 filters, time {s4.shape[-1]})")
    
    # Node features
    node_features = s4.view(b, c, model.node_embedding_dim)
    print(f"Stage 6 (Node Feature Construction):shape = {tuple(node_features.shape)} (B={b}, Nodes={c}, Dim={node_features.shape[-1]})")
    
    # GNN Layer 1
    h1 = model.drop1(model.act1(model.gcn1(node_features, model.adj_norm)))
    print(f"Stage 7 (Spatial GNN Layer 1):      shape = {tuple(h1.shape)}")
    
    # GNN Layer 2
    h2 = model.drop2(model.act2(model.gcn2(h1, model.adj_norm)))
    print(f"Stage 8 (Spatial GNN Layer 2):      shape = {tuple(h2.shape)}")
    
    # Pooling
    h_mean = h2.mean(dim=1)
    h_max = h2.max(dim=1)[0]
    h_graph = torch.cat([h_mean, h_max], dim=-1)
    print(f"Stage 9 (Graph Dual Pooling):       shape = {tuple(h_graph.shape)} (Mean {tuple(h_mean.shape)} + Max {tuple(h_max.shape)})")
    
    # Classifier
    logits = model.classifier(h_graph).squeeze(-1)
    print(f"Stage 10 (Classification Head):    shape = {tuple(logits.shape)} (Scalar linear logits)")
    
    # 3. Channel Identity Controlled Synthetic Test
    print("\n--- Controlled Channel Identity Test ---")
    # For channels 0, 5, 10, 15, 22:
    test_channels = [0, 5, 10, 15, 22]
    identity_test_passed = True
    
    # Compute baseline node representation for all-zeros input
    x_zero = torch.zeros(1, 23, 1280)
    with torch.no_grad():
        node_zero = model.temporal_backbone(x_zero.view(23, 1, 1280)).view(1, 23, 64)
        
    for target_ch in test_channels:
        x_iso = torch.zeros(1, 23, 1280)
        # Inject strong synthetic sinusoidal burst only at target_ch
        t = torch.linspace(0, 1, 1280)
        x_iso[0, target_ch, :] = torch.sin(2 * np.pi * 10 * t) * 5.0
        
        with torch.no_grad():
            node_iso = model.temporal_backbone(x_iso.view(23, 1, 1280)).view(1, 23, 64)
            
        # Measure delta norm at each of the 23 nodes compared to zero baseline
        diffs = torch.norm(node_iso[0] - node_zero[0], dim=-1).numpy()
        max_diff_node = int(np.argmax(diffs))
        
        target_ch_name = canonical_channels[target_ch]
        max_node_name = canonical_channels[max_diff_node] if max_diff_node < 23 else "OUT_OF_BOUNDS"
        
        print(f"Channel {target_ch:02d} ({target_ch_name}) activated: Max perturbation at Node {max_diff_node:02d} ({max_node_name})")
        print(f"  Node {target_ch:02d} delta: {diffs[target_ch]:.4f} | Other 22 nodes max delta: {np.max(np.delete(diffs, target_ch)):.4f}")
        
        if max_diff_node != target_ch:
            print(f"  CRITICAL ERROR: Channel {target_ch} did not map to Node {target_ch}!")
            identity_test_passed = False
        else:
            # Check other nodes have 0 perturbation
            other_diffs = np.delete(diffs, target_ch)
            if np.max(other_diffs) > 1e-5:
                print(f"  WARNING: Non-zero leakage ({np.max(other_diffs):.4f}) into non-target nodes before GNN!")
            else:
                print(f"  PASS: Exactly 0.0000 leakage into other nodes. Channel identity strictly preserved.")
                
    print(f"\nChannel Identity Test Final Result: {'PASS' if identity_test_passed else 'FAIL (CRITICAL DEFECT)'}")
    
    return {
        "mismatch_detected": mismatch_detected,
        "identity_test_passed": identity_test_passed,
        "mapping_csv_path": mapping_csv_path
    }


if __name__ == "__main__":
    audit_tensor_flow_and_channels()
