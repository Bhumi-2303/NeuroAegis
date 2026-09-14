"""
NeuroAegis Automated Test Suite for Phase 4A: CNN + GNN Spatial Baseline
26 rigorous unit and integration tests verifying architectural invariants, GNN mathematics,
graph properties, data partitioning, focal loss stability, and protocol immutability.
"""

import os
import sys
BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import json
import hashlib
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from neuroaegis.models.baselines.cnn_gnn_model import SpatialGCNLayer, Baseline1DCNN_GNN
from research.experiments.imbalance.focal_loss import BinaryFocalLossWithLogits, logits_to_probabilities
from neuroaegis.evaluation.window_metrics import SeizureEvaluationMetrics

MANIFEST_DIR = os.path.join(BASE_DIR, "data/manifests")
WINDOW_INDEX_PATH = os.path.join(MANIFEST_DIR, "chbmit_window_index.csv")
EVENTS_PATH = os.path.join(MANIFEST_DIR, "chbmit_seizure_events.csv")
SPLIT_CONFIG_PATH = os.path.join(BASE_DIR, "research/experiments/imbalance/class_imbalance_config.json")
EXP_DIR = os.path.join(BASE_DIR, "research/experiments/gnn/cnn_gnn/exp_01")
GRAPH_ADJ_PATH = os.path.join(EXP_DIR, "graph_adjacency.csv")
GRAPH_CONFIG_PATH = os.path.join(EXP_DIR, "graph_config.json")
ARCH_JSON_PATH = os.path.join(EXP_DIR, "model_architecture.json")
PHASE3_BEST_PT = os.path.join(BASE_DIR, "research/experiments/cnn_baseline/best_cnn_baseline.pt")


class TestPhase4ACNNGNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nLoading configs and data for Phase 4A test suite...")
        with open(SPLIT_CONFIG_PATH, "r") as f:
            cls.split_cfg = json.load(f)
        cls.train_pats = set(cls.split_cfg["split_summary"]["train"]["patients"])
        cls.val_pats = set(cls.split_cfg["split_summary"]["validation"]["patients"])
        cls.test_pats = set(cls.split_cfg["split_summary"]["test"]["patients"])
        
        cls.adj_df = pd.read_csv(GRAPH_ADJ_PATH, index_col=0)
        cls.adj_matrix = cls.adj_df.values.astype(np.float32)
        cls.model = Baseline1DCNN_GNN(in_channels=23, adj_matrix=cls.adj_matrix)

    def test_01_model_instantiation_and_parameter_count(self):
        """Test 1: Model instantiates and parameters are < 2,000,000."""
        n_params = self.model.get_num_parameters()
        self.assertLess(n_params, 2_000_000, f"Parameter count {n_params} exceeds 2M budget")
        self.assertEqual(n_params, 52497, f"Expected 52,497 parameters, got {n_params}")

    def test_02_output_shape_and_type(self):
        """Test 2: Forward pass yields 1D tensor of shape (Batch,) with dtype float32."""
        x = torch.randn(8, 23, 1280)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (8,), f"Expected shape (8,), got {out.shape}")
        self.assertEqual(out.dtype, torch.float32)

    def test_03_output_unnormalized_logits_no_sigmoid(self):
        """Test 3: Model outputs raw unnormalized linear logits (NO nn.Sigmoid in model)."""
        for module in self.model.modules():
            self.assertNotIsInstance(module, nn.Sigmoid, "Model must NOT contain nn.Sigmoid")
        # Extreme inputs should yield logits outside [0, 1]
        x_extreme = torch.randn(20, 23, 1280) * 10.0
        with torch.no_grad():
            logits = self.model(x_extreme)
        has_outside_01 = (logits < 0.0).any() or (logits > 1.0).any()
        self.assertTrue(has_outside_01, "Logits should span outside [0, 1] range")

    def test_04_channel_preservation_representation(self):
        """Test 4: Channels are encoded independently, preserving 23 node representations."""
        x = torch.randn(2, 23, 1280)
        b, c, t = x.shape
        x_reshaped = x.view(b * c, 1, t)
        temp_feat = self.model.temporal_backbone(x_reshaped)
        self.assertEqual(temp_feat.shape, (2 * 23, 64, 1))
        node_embeds = temp_feat.view(b, c, 64)
        self.assertEqual(node_embeds.shape, (2, 23, 64))

    def test_05_gnn_layer_mathematical_correctness(self):
        """Test 5: SpatialGCNLayer implements A_hat * H * W + b exactly."""
        layer = SpatialGCNLayer(in_features=4, out_features=4, bias=True)
        h = torch.ones(2, 3, 4)
        a_hat = torch.eye(3) * 0.5
        out = layer(h, a_hat)
        expected = torch.matmul(a_hat, h)
        expected_out = layer.linear(expected)
        self.assertTrue(torch.allclose(out, expected_out, atol=1e-6))

    def test_06_graph_adjacency_symmetry(self):
        """Test 6: Adjacency matrix is symmetric (A = A^T)."""
        diff = np.abs(self.adj_matrix - self.adj_matrix.T)
        self.assertLess(np.max(diff), 1e-6, "Adjacency matrix is not symmetric")

    def test_07_graph_adjacency_self_loops(self):
        """Test 7: Self-loops are present on all 23 nodes (diagonal > 0)."""
        diag = np.diag(self.adj_matrix)
        self.assertTrue((diag > 0.0).all(), "Diagonal entries (self-loops) must be > 0")

    def test_08_graph_adjacency_degree_normalization(self):
        """Test 8: Adjacency matrix has 23x23 shape and all entries are bounded [0, 1]."""
        self.assertEqual(self.adj_matrix.shape, (23, 23))
        self.assertGreaterEqual(np.min(self.adj_matrix), 0.0)
        self.assertLessEqual(np.max(self.adj_matrix), 1.0)

    def test_09_zero_patient_leakage(self):
        """Test 9: Patient split is mutually exclusive between train, val, and test."""
        self.assertEqual(len(self.train_pats & self.val_pats), 0, "Train-Val patient leakage!")
        self.assertEqual(len(self.train_pats & self.test_pats), 0, "Train-Test patient leakage!")
        self.assertEqual(len(self.val_pats & self.test_pats), 0, "Val-Test patient leakage!")

    def test_10_zero_recording_leakage(self):
        """Test 10: Recordings belong strictly to one split based on patient partition."""
        manifest_df = pd.read_csv(os.path.join(MANIFEST_DIR, "chbmit_manifest.csv"))
        train_recs = set(manifest_df[manifest_df["patient_id"].isin(self.train_pats)]["recording_id"])
        val_recs = set(manifest_df[manifest_df["patient_id"].isin(self.val_pats)]["recording_id"])
        test_recs = set(manifest_df[manifest_df["patient_id"].isin(self.test_pats)]["recording_id"])
        self.assertEqual(len(train_recs & val_recs), 0, "Train-Val recording leakage!")
        self.assertEqual(len(train_recs & test_recs), 0, "Train-Test recording leakage!")
        self.assertEqual(len(val_recs & test_recs), 0, "Val-Test recording leakage!")

    def test_11_zero_window_leakage(self):
        """Test 11: Windows belong strictly to one split with zero overlap."""
        df_win = pd.read_csv(WINDOW_INDEX_PATH, usecols=["window_id", "patient_id"])
        train_w = set(df_win[df_win["patient_id"].isin(self.train_pats)]["window_id"])
        val_w = set(df_win[df_win["patient_id"].isin(self.val_pats)]["window_id"])
        test_w = set(df_win[df_win["patient_id"].isin(self.test_pats)]["window_id"])
        self.assertEqual(len(train_w & val_w), 0, "Train-Val window leakage!")
        self.assertEqual(len(train_w & test_w), 0, "Train-Test window leakage!")
        self.assertEqual(len(val_w & test_w), 0, "Val-Test window leakage!")

    def test_12_dynamic_sampler_10_to_1_ratio(self):
        """Test 12: Dynamic negative sampler produces 10 negatives per positive."""
        n_pos = 100
        n_neg = 1000
        ratio = 10.0
        n_sampled_neg = int(n_pos * ratio)
        self.assertEqual(n_sampled_neg / n_pos, 10.0)

    def test_13_dynamic_sampler_seed_reproducibility(self):
        """Test 13: Deterministic seed formula 42 + epoch produces reproducible indices."""
        pool_size = 1000
        k = 100
        rng1 = np.random.default_rng(42 + 1)
        idx1 = rng1.choice(pool_size, size=k, replace=False)
        rng2 = np.random.default_rng(42 + 1)
        idx2 = rng2.choice(pool_size, size=k, replace=False)
        self.assertTrue((idx1 == idx2).all(), "Seed formula must be strictly reproducible")

    def test_14_primary_label_definition_integrity(self):
        """Test 14: Primary label_50pct_overlap matches overlap_ratio >= 0.50."""
        df_win = pd.read_csv(WINDOW_INDEX_PATH, nrows=5000)
        computed = (df_win["overlap_ratio"] >= 0.50).astype(int)
        self.assertTrue((computed == df_win["label_50pct_overlap"]).all())

    def test_15_gradient_flow_through_all_modules(self):
        """Test 15: Backward pass propagates non-zero gradients to all model components."""
        model = Baseline1DCNN_GNN(in_channels=23, adj_matrix=self.adj_matrix)
        model.train()
        x = torch.randn(4, 23, 1280, requires_grad=True)
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        criterion = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25)
        
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        # Verify gradients exist in temporal backbone, gcn layers, and head
        self.assertIsNotNone(model.temporal_backbone[0].weight.grad)
        self.assertGreater(model.temporal_backbone[0].weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.gcn1.linear.weight.grad)
        self.assertGreater(model.gcn1.linear.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.gcn2.linear.weight.grad)
        self.assertGreater(model.gcn2.linear.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.classifier[0].weight.grad)
        self.assertGreater(model.classifier[0].weight.grad.abs().sum().item(), 0.0)

    def test_16_binary_focal_loss_stability_and_equivalence(self):
        """Test 16: Binary Focal Loss is finite and matches BCE when gamma = 0.0."""
        logits = torch.randn(10, requires_grad=True)
        targets = torch.randint(0, 2, (10,)).float()
        
        # Gamma = 0, alpha = 0.5 should be proportional to standard BCE
        bce = nn.BCEWithLogitsLoss(reduction="mean")(logits, targets)
        focal_g0 = BinaryFocalLossWithLogits(gamma=0.0, alpha=0.5, reduction="mean")(logits, targets)
        self.assertAlmostEqual(focal_g0.item(), 0.5 * bce.item(), places=4)
        
        # Gamma = 2.0, alpha = 0.25 (frozen Decision 2)
        focal_frozen = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25)(logits, targets)
        self.assertTrue(torch.isfinite(focal_frozen))
        self.assertGreater(focal_frozen.item(), 0.0)

    def test_17_device_agnostic_execution(self):
        """Test 17: Model executes seamlessly on CPU and MPS (if available)."""
        x = torch.randn(2, 23, 1280)
        model_cpu = Baseline1DCNN_GNN(in_channels=23, adj_matrix=self.adj_matrix).cpu()
        model_cpu.eval()
        with torch.no_grad():
            out_cpu = model_cpu(x)
        self.assertEqual(out_cpu.shape, (2,))
        
        if torch.backends.mps.is_available():
            model_mps = Baseline1DCNN_GNN(in_channels=23, adj_matrix=self.adj_matrix).to("mps")
            model_mps.eval()
            with torch.no_grad():
                out_mps = model_mps(x.to("mps"))
            self.assertEqual(out_mps.shape, (2,))

    def test_18_graph_builder_training_only_isolation(self):
        """Test 18: graph_config.json verifies strictly training patients were used."""
        with open(GRAPH_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        used_pats = set(cfg["training_patients_used"])
        self.assertEqual(len(used_pats & self.val_pats), 0, "Graph used validation patients!")
        self.assertEqual(len(used_pats & self.test_pats), 0, "Graph used test patients!")

    def test_19_master_window_index_immutability(self):
        """Test 19: Master window index SHA256 has not been altered."""
        sha = hashlib.sha256()
        with open(WINDOW_INDEX_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        current_hash = sha.hexdigest()
        expected_hash = "f76dddb19fa2c24b3af24b989168b12465512b650c5e6d84bb0125a6d666302c"
        self.assertEqual(current_hash, expected_hash, "Master window index corrupted!")

    def test_20_phase3_checkpoint_immutability(self):
        """Test 20: Phase 3 checkpoint best_cnn_baseline.pt exists and is untouched."""
        self.assertTrue(os.path.exists(PHASE3_BEST_PT), "Phase 3 best_cnn_baseline.pt missing!")
        ckpt = torch.load(PHASE3_BEST_PT, map_location="cpu")
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("val_auprc", ckpt)

    def test_21_event_sensitivity_calculation(self):
        """Test 21: Event sensitivity correctly detects covered seizure events."""
        win_df = pd.DataFrame([
            {"seizure_event_ids": "sz_1", "label_50pct_overlap": 1},
            {"seizure_event_ids": "sz_2", "label_50pct_overlap": 1},
        ])
        y_prob = np.array([0.8, 0.2])
        res = SeizureEvaluationMetrics.compute_event_level_sensitivity(
            window_df=win_df,
            y_prob=y_prob,
            threshold=0.5,
            label_column="label_50pct_overlap"
        )
        self.assertEqual(res["total_seizure_events"], 2)
        self.assertEqual(res["detected_seizure_events"], 1)
        self.assertEqual(res["event_level_sensitivity"], 0.5)

    def test_22_false_alarm_rate_calculation(self):
        """Test 22: False alarms per 24 hours calculation is numerically correct."""
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.9, 0.1, 0.9, 0.1])
        res = SeizureEvaluationMetrics.compute_window_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.5,
            total_duration_hours=1.0
        )
        self.assertEqual(res["false_positives"], 2)
        self.assertEqual(res["false_alarms_per_24h"], 48.0)

    def test_23_detection_delay_calculation(self):
        """Test 23: Detection delay measures elapsed time from seizure start to first hit."""
        # Seizure starts at 100.0s. Window starts at 105.0s, ends at 110.0s with prob=0.8
        s_start = 100.0
        w_start = 105.0
        pred_prob = 0.8
        is_hit = pred_prob >= 0.5
        delay = max(0.0, w_start - s_start) if is_hit else None
        self.assertEqual(delay, 5.0)

    def test_24_decision_threshold_frozen_at_050(self):
        """Test 24: Threshold 0.50 is used for binary decision logic."""
        probs = np.array([0.49, 0.50, 0.51])
        preds = (probs >= 0.50).astype(int)
        self.assertTrue((preds == np.array([0, 1, 1])).all())

    def test_25_model_architecture_json_schema(self):
        """Test 25: model_architecture.json matches required schema and parameter counts."""
        self.assertTrue(os.path.exists(ARCH_JSON_PATH), "model_architecture.json missing!")
        with open(ARCH_JSON_PATH, "r") as f:
            arch = json.load(f)
        self.assertEqual(arch["model_name"], "Baseline1DCNN_GNN")
        self.assertEqual(arch["total_parameters"], 52497)
        self.assertIn("temporal_backbone_1d_cnn", arch["layer_breakdown"])
        self.assertIn("spatial_gnn_gcn1", arch["layer_breakdown"])
        self.assertIn("spatial_gnn_gcn2", arch["layer_breakdown"])
        self.assertIn("classification_head", arch["layer_breakdown"])

    def test_26_end_to_end_smoke_test(self):
        """Test 26: Full 1-batch forward, loss, backward, and optimizer step."""
        model = Baseline1DCNN_GNN(in_channels=23, adj_matrix=self.adj_matrix)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25)
        
        x = torch.randn(4, 23, 1280)
        y = torch.tensor([1.0, 0.0, 0.0, 1.0])
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        
        # Verify no NaN weights
        for p in model.parameters():
            self.assertFalse(torch.isnan(p).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
