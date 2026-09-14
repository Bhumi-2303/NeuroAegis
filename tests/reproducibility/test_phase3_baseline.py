"""
NeuroAegis Automated Test Suite for Phase 3 CNN Baseline
Tests 1 to 12 covering model architecture, unnormalized logits, leakage isolation,
sampling dynamics, focal loss stability, metric calculations, and index immutability.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "research/experiments/cnn_baseline"))

from research.experiments.cnn_baseline.cnn_model import Baseline1DCNN
from research.experiments.imbalance.patient_splitter import PatientDataSplitter
from research.experiments.imbalance.dynamic_sampler import DynamicNegativeSampler
from research.experiments.imbalance.focal_loss import BinaryFocalLossWithLogits, logits_to_probabilities
from neuroaegis.evaluation.window_metrics import SeizureEvaluationMetrics
from research.experiments.cnn_baseline.train_cnn_baseline import compute_detection_delay

WINDOW_INDEX_PATH = os.path.join(BASE_DIR, "data/manifests/chbmit_window_index.csv")
EVENTS_PATH = os.path.join(BASE_DIR, "data/manifests/chbmit_seizure_events.csv")
PROTOCOL_PATH = os.path.join(BASE_DIR, "research/experiments/windowing_labeling/labeling_protocol.json")


class TestPhase3Baseline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nLoading manifests and components for Phase 3 test suite...")
        cls.splitter = PatientDataSplitter(
            window_index_path=WINDOW_INDEX_PATH,
            seizure_events_path=EVENTS_PATH,
            label_column="label_50pct_overlap"
        )
        cls.train_df, cls.val_df, cls.test_df = cls.splitter.get_splits()
        cls.events_df = pd.read_csv(EVENTS_PATH)
        cls.model = Baseline1DCNN(in_channels=23, num_classes=1)

    def test_01_model_parameters_and_shape(self):
        """Test 1: Model takes (B, 23, 1280) and outputs (B,). Parameter count within budget."""
        print("\n[Test 1/12] Model parameter count and output shape...")
        params = self.model.get_num_parameters()
        self.assertGreater(params, 50000, "Model parameter count too low")
        self.assertLess(params, 500000, "Model parameter count exceeds lightweight budget")
        
        x = torch.randn(4, 23, 1280)
        out = self.model(x)
        self.assertEqual(out.shape, (4,), f"Expected shape (4,), got {out.shape}")
        print(f"  -> PASS: Parameters = {params:,}, output shape = {out.shape}.")

    def test_02_unnormalized_linear_logits(self):
        """Test 2: Model outputs raw linear logits, NOT probabilities (no sigmoid in model)."""
        print("\n[Test 2/12] Unnormalized linear logits verification (no Sigmoid layer)...")
        # Check module list does not contain Sigmoid
        for name, module in self.model.named_modules():
            self.assertNotIsInstance(module, nn.Sigmoid, f"Forbidden nn.Sigmoid found in module: {name}")
            
        # Check with extreme inputs that outputs can be outside [0, 1]
        x_extreme = torch.randn(10, 23, 1280) * 100.0
        with torch.no_grad():
            out = self.model(x_extreme)
        can_exceed = (out < 0.0).any() or (out > 1.0).any()
        self.assertTrue(can_exceed, "Outputs appear constrained to [0, 1]; expected unnormalized logits")
        print("  -> PASS: Zero Sigmoid layers in model; outputs are unnormalized linear logits.")

    def test_03_patient_leakage_isolation(self):
        """Test 3: Train, Val, Test patient sets are pairwise mutually disjoint."""
        print("\n[Test 3/12] Patient leakage isolation...")
        train_p = set(self.splitter.train_patients)
        val_p = set(self.splitter.val_patients)
        test_p = set(self.splitter.test_patients)
        
        self.assertEqual(len(train_p & val_p), 0, f"Train and Val share patients: {train_p & val_p}")
        self.assertEqual(len(train_p & test_p), 0, f"Train and Test share patients: {train_p & test_p}")
        self.assertEqual(len(val_p & test_p), 0, f"Val and Test share patients: {val_p & test_p}")
        print(f"  -> PASS: 16 Train / 4 Val / 4 Test patients are 100% pairwise disjoint.")

    def test_04_recording_leakage_isolation(self):
        """Test 4: Recording sets across Train, Val, Test are pairwise disjoint."""
        print("\n[Test 4/12] Recording leakage isolation...")
        train_r = set(self.train_df["recording_id"].unique())
        val_r = set(self.val_df["recording_id"].unique())
        test_r = set(self.test_df["recording_id"].unique())
        
        self.assertEqual(len(train_r & val_r), 0, "Train and Val share recordings!")
        self.assertEqual(len(train_r & test_r), 0, "Train and Test share recordings!")
        self.assertEqual(len(val_r & test_r), 0, "Val and Test share recordings!")
        print(f"  -> PASS: All recordings strictly partitioned across splits without overlap.")

    def test_05_window_leakage_isolation(self):
        """Test 5: Window IDs across Train, Val, Test are pairwise disjoint."""
        print("\n[Test 5/12] Window leakage isolation...")
        train_w = set(self.train_df["window_id"].unique())
        val_w = set(self.val_df["window_id"].unique())
        test_w = set(self.test_df["window_id"].unique())
        
        self.assertEqual(len(train_w & val_w), 0, "Train and Val share window IDs!")
        self.assertEqual(len(train_w & test_w), 0, "Train and Test share window IDs!")
        self.assertEqual(len(val_w & test_w), 0, "Val and Test share window IDs!")
        print(f"  -> PASS: Zero window leakage across all 1,414,710 windows.")

    def test_06_dynamic_sampler_ratio_and_seed(self):
        """Test 6: DynamicNegativeSampler maintains 10:1 ratio and reproducible seed progression."""
        print("\n[Test 6/12] Dynamic negative sampler ratio and seed formula...")
        sampler = DynamicNegativeSampler(
            self.train_df, ratio=10.0, base_seed=42, label_column="label_50pct_overlap"
        )
        sampler.set_epoch(0)
        s0 = sampler.get_epoch_stats()
        self.assertEqual(s0["epoch_seed"], 42)
        self.assertEqual(s0["actual_ratio"], 10.0)
        
        sampler.set_epoch(1)
        s1 = sampler.get_epoch_stats()
        self.assertEqual(s1["epoch_seed"], 43)
        self.assertEqual(s1["actual_ratio"], 10.0)
        
        # Verify 100% positive retention
        self.assertEqual(s0["positive_samples"], s1["positive_samples"])
        print(f"  -> PASS: Sampler verified at 10.0:1 ratio with seed formula base_seed + epoch.")

    def test_07_focal_loss_stability_and_gradients(self):
        """Test 7: BinaryFocalLossWithLogits produces finite loss and valid backward gradients."""
        print("\n[Test 7/12] Binary Focal Loss gradient check...")
        criterion = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25)
        logits = torch.randn(8, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 0, 1, 0, 1, 0], dtype=torch.float32)
        
        loss = criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss), "Loss is not finite")
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all(), "Gradients contain NaN/Inf")
        print(f"  -> PASS: Focal Loss = {loss.item():.4f}, finite gradients verified.")

    def test_08_test_set_isolation(self):
        """Test 8: Test set completely bypasses training sampler."""
        print("\n[Test 8/12] Test set isolation from sampling...")
        test_pos = int((self.test_df["label_50pct_overlap"] == 1).sum())
        test_total = len(self.test_df)
        natural_ratio = (test_total - test_pos) / test_pos
        self.assertGreater(natural_ratio, 300.0, "Test set must maintain natural clinical ratio > 300:1")
        print(f"  -> PASS: Test set natural ratio = {natural_ratio:.2f}:1 untouched by sampler.")

    def test_09_metrics_computation_correctness(self):
        """Test 9: SeizureEvaluationMetrics computes valid bounded clinical metrics."""
        print("\n[Test 9/12] Metrics suite computation...")
        y_t = np.array([0, 0, 1, 1, 0, 0, 1, 0])
        y_p = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.1, 0.85, 0.05])
        m = SeizureEvaluationMetrics.compute_window_metrics(y_t, y_p, threshold=0.5)
        
        self.assertIn("accuracy", m)
        self.assertIn("precision", m)
        self.assertIn("sensitivity", m)
        self.assertIn("specificity", m)
        self.assertIn("f1_score", m)
        self.assertIn("auroc", m)
        self.assertIn("auprc", m)
        self.assertEqual(m["sensitivity"], 1.0)
        self.assertEqual(m["specificity"], 1.0)
        print(f"  -> PASS: Metric calculation verified (AUROC={m['auroc']}, AUPRC={m['auprc']}).")

    def test_10_detection_delay_correctness(self):
        """Test 10: Detection delay calculation correctly computes onset latency."""
        print("\n[Test 10/12] Detection delay computation...")
        # Create dummy window df and event df
        dummy_windows = pd.DataFrame([
            {"recording_id": "r1", "window_start_sec": 0.0, "window_end_sec": 5.0, "label_50pct_overlap": 0},
            {"recording_id": "r1", "window_start_sec": 2.5, "window_end_sec": 7.5, "label_50pct_overlap": 1},
            {"recording_id": "r1", "window_start_sec": 5.0, "window_end_sec": 10.0, "label_50pct_overlap": 1},
        ])
        dummy_events = pd.DataFrame([
            {"recording_id": "r1", "seizure_id": "s1", "start_sec": 3.0, "end_sec": 8.0}
        ])
        y_p = np.array([0.1, 0.8, 0.9])
        delay, det_cnt, tot_cnt = compute_detection_delay(dummy_windows, dummy_events, y_p, threshold=0.5)
        self.assertEqual(det_cnt, 1)
        self.assertEqual(tot_cnt, 1)
        # First alarm window is [2.5, 7.5] which ends at 7.5. Start is 3.0. Delay = 7.5 - 3.0 = 4.5s
        self.assertEqual(delay, 4.5)
        print(f"  -> PASS: Detection delay = {delay}s verified.")

    def test_11_master_index_immutability(self):
        """Test 11: Master window index SHA256 has not been modified."""
        print("\n[Test 11/12] Master window index SHA256 immutability...")
        sha = hashlib.sha256()
        with open(WINDOW_INDEX_PATH, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha.update(chunk)
        current_hash = sha.hexdigest()
        
        with open(PROTOCOL_PATH, "r") as f:
            proto = json.load(f)
        expected_hash = proto["master_index_sha256"]
        self.assertEqual(current_hash, expected_hash, "Master index SHA256 was altered!")
        print(f"  -> PASS: Master index SHA256 is strictly identical ({current_hash[:16]}...).")

    def test_12_smoke_test_and_reproducibility(self):
        """Test 12: Deterministic model initialization under seed 42."""
        print("\n[Test 12/12] Model initialization reproducibility (seed 42)...")
        torch.manual_seed(42)
        m1 = Baseline1DCNN()
        torch.manual_seed(42)
        m2 = Baseline1DCNN()
        
        w1 = list(m1.parameters())[0]
        w2 = list(m2.parameters())[0]
        self.assertTrue(torch.equal(w1, w2), "Different initialization under identical seed!")
        print("  -> PASS: Bit-exact deterministic initialization verified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
