"""
NeuroAegis Automated Test Suite for Phase 2 Labeling Strategy Audit
Tests 1 to 12 covering label definitions, subset invariants, event coverage,
short seizures, overlap bounds, unique IDs, index immutability, and protocol integrity.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
MANIFEST_DIR = os.path.join(BASE_DIR, "research/data/manifests")
PHASE2_DIR = os.path.join(BASE_DIR, "research/phase_2")
WINDOW_INDEX_PATH = os.path.join(MANIFEST_DIR, "chbmit_window_index.csv")
EVENTS_PATH = os.path.join(MANIFEST_DIR, "chbmit_seizure_events.csv")
PROTOCOL_PATH = os.path.join(PHASE2_DIR, "labeling_protocol.json")


class TestLabelingStrategy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nLoading manifests for labeling strategy test suite...")
        cls.df_windows = pd.read_csv(WINDOW_INDEX_PATH, low_memory=False)
        cls.df_events = pd.read_csv(EVENTS_PATH)
        print(f"Loaded {len(cls.df_windows):,} windows, {len(cls.df_events)} events.")

    def test_01_strategy_a_definition(self):
        """Test 1: Strategy A label_any_overlap == (overlap_duration_sec > 0)."""
        print("\n[Test 1/12] Strategy A definition correctness...")
        computed = (self.df_windows["overlap_duration_sec"] > 0.0).astype(int)
        self.assertTrue((computed == self.df_windows["label_any_overlap"]).all(),
                        "Strategy A definition mismatch")
        print("  -> PASS: Strategy A definition is correct.")

    def test_02_strategy_b_definition(self):
        """Test 2: Strategy B label_50pct_overlap == (overlap_ratio >= 0.5)."""
        print("\n[Test 2/12] Strategy B definition correctness...")
        computed = (self.df_windows["overlap_ratio"] >= 0.50).astype(int)
        self.assertTrue((computed == self.df_windows["label_50pct_overlap"]).all(),
                        "Strategy B definition mismatch")
        print("  -> PASS: Strategy B definition is correct.")

    def test_03_b_subset_of_a(self):
        """Test 3: Strategy B positive set is a subset of Strategy A."""
        print("\n[Test 3/12] Subset invariant (B subset of A)...")
        pos_a = int(self.df_windows["label_any_overlap"].sum())
        pos_b = int(self.df_windows["label_50pct_overlap"].sum())
        self.assertGreaterEqual(pos_a, pos_b, "Strategy B positives exceed Strategy A")
        # Check every B=1 has A=1
        b_pos_mask = self.df_windows["label_50pct_overlap"] == 1
        a_at_b_pos = self.df_windows.loc[b_pos_mask, "label_any_overlap"]
        self.assertTrue((a_at_b_pos == 1).all(), "Found B=1 where A=0")
        print(f"  -> PASS: pos_B ({pos_b}) <= pos_A ({pos_a}), every B=1 has A=1.")

    def test_04_no_a0_b1_windows(self):
        """Test 4: No impossible A=0/B=1 windows exist."""
        print("\n[Test 4/12] No impossible A=0/B=1 windows...")
        violations = self.df_windows[
            (self.df_windows["label_any_overlap"] == 0) &
            (self.df_windows["label_50pct_overlap"] == 1)
        ]
        self.assertEqual(len(violations), 0,
                         f"Found {len(violations)} A=0/B=1 violations!")
        print("  -> PASS: Zero A=0/B=1 windows. Subset invariant holds.")

    def test_05_all_198_events_in_audit(self):
        """Test 5: All 198 seizure events are present."""
        print("\n[Test 5/12] All 198 events present...")
        self.assertEqual(len(self.df_events), 198,
                         f"Expected 198 events, found {len(self.df_events)}")
        print("  -> PASS: All 198 seizure events confirmed.")

    def test_06_short_seizure_coverage(self):
        """Test 6: Short seizure events (chb16_17, chb16_16) are correctly covered."""
        print("\n[Test 6/12] Short seizure event coverage...")
        # chb16_17 seizure 2: 1694s-1700s (6s)
        w_17 = self.df_windows[
            (self.df_windows["recording_id"] == "chb16_17") &
            (self.df_windows["window_end_sec"] > 1694.0) &
            (self.df_windows["window_start_sec"] < 1700.0)
        ]
        pos_a_17 = int(w_17["label_any_overlap"].sum())
        pos_b_17 = int(w_17["label_50pct_overlap"].sum())
        self.assertGreaterEqual(pos_a_17, 1, "chb16_17 6s seizure must have A >= 1")
        self.assertGreaterEqual(pos_b_17, 1, "chb16_17 6s seizure must have B >= 1")

        # chb16_16 seizure 1: 1214s-1220s (6s)
        w_16 = self.df_windows[
            (self.df_windows["recording_id"] == "chb16_16") &
            (self.df_windows["window_end_sec"] > 1214.0) &
            (self.df_windows["window_start_sec"] < 1220.0)
        ]
        pos_a_16 = int(w_16["label_any_overlap"].sum())
        pos_b_16 = int(w_16["label_50pct_overlap"].sum())
        self.assertGreaterEqual(pos_a_16, 1, "chb16_16 6s seizure must have A >= 1")
        self.assertGreaterEqual(pos_b_16, 1, "chb16_16 6s seizure must have B >= 1")
        print(f"  -> PASS: chb16_17 A={pos_a_17}/B={pos_b_17}, chb16_16 A={pos_a_16}/B={pos_b_16}.")

    def test_07_overlap_ratios_bounded(self):
        """Test 7: Overlap ratios are in [0, 1]."""
        print("\n[Test 7/12] Overlap ratio bounds [0, 1]...")
        min_r = self.df_windows["overlap_ratio"].min()
        max_r = self.df_windows["overlap_ratio"].max()
        self.assertGreaterEqual(min_r, 0.0, f"Min overlap ratio {min_r} < 0")
        self.assertLessEqual(max_r, 1.0, f"Max overlap ratio {max_r} > 1")
        print(f"  -> PASS: All overlap ratios in [{min_r}, {max_r}] within [0, 1].")

    def test_08_unique_window_ids(self):
        """Test 8: No duplicate window IDs."""
        print("\n[Test 8/12] Unique window IDs...")
        total = len(self.df_windows)
        unique = self.df_windows["window_id"].nunique()
        self.assertEqual(total, unique,
                         f"Found {total - unique} duplicate window IDs")
        print(f"  -> PASS: All {unique:,} window IDs are unique.")

    def test_09_master_index_unchanged(self):
        """Test 9: Master index file hash matches expected value."""
        print("\n[Test 9/12] Master index immutability (SHA256)...")
        sha = hashlib.sha256()
        with open(WINDOW_INDEX_PATH, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha.update(chunk)
        current_hash = sha.hexdigest()
        # Verify the protocol file records the same hash
        if os.path.exists(PROTOCOL_PATH):
            with open(PROTOCOL_PATH, "r") as f:
                protocol = json.load(f)
            expected_hash = protocol.get("master_index_sha256", "")
            self.assertEqual(current_hash, expected_hash,
                             f"Master index hash mismatch! Current: {current_hash}, Protocol: {expected_hash}")
        print(f"  -> PASS: Master index SHA256 = {current_hash[:32]}...")

    def test_10_primary_strategy_from_protocol(self):
        """Test 10: Primary strategy is correctly loaded from labeling_protocol.json."""
        print("\n[Test 10/12] Primary strategy from frozen protocol...")
        self.assertTrue(os.path.exists(PROTOCOL_PATH),
                        "labeling_protocol.json does not exist")
        with open(PROTOCOL_PATH, "r") as f:
            protocol = json.load(f)
        self.assertIn("primary_label_strategy", protocol,
                      "Missing primary_label_strategy key")
        self.assertIn("definition", protocol,
                      "Missing definition key")
        self.assertIn("overlap_threshold", protocol,
                      "Missing overlap_threshold key")
        self.assertEqual(protocol["window_duration_sec"], 5.0,
                         "Window duration mismatch")
        self.assertEqual(protocol["window_stride_sec"], 2.5,
                         "Window stride mismatch")
        self.assertEqual(protocol["sampling_frequency_hz"], 256.0,
                         "Sampling frequency mismatch")
        self.assertEqual(protocol["dataset"], "CHB-MIT",
                         "Dataset mismatch")
        print(f"  -> PASS: Protocol loaded: {protocol['primary_label_strategy']}.")

    def test_11_no_model_training(self):
        """Test 11: No model training was executed in this audit."""
        print("\n[Test 11/12] No model training verification...")
        # Check that no model checkpoint, training log, or weights file exists in phase_2
        model_extensions = [".pt", ".pth", ".h5", ".ckpt", ".pb", ".onnx"]
        found_models = []
        for root, dirs, files in os.walk(PHASE2_DIR):
            for f in files:
                if any(f.endswith(ext) for ext in model_extensions):
                    found_models.append(os.path.join(root, f))
        self.assertEqual(len(found_models), 0,
                         f"Model files found in phase_2: {found_models}")
        print("  -> PASS: No model checkpoints or weights found in phase_2.")

    def test_12_no_model_performance_in_decision(self):
        """Test 12: The protocol rationale does not reference model performance metrics."""
        print("\n[Test 12/12] No model performance in decision rationale...")
        if os.path.exists(PROTOCOL_PATH):
            with open(PROTOCOL_PATH, "r") as f:
                protocol = json.load(f)
            rationale = protocol.get("rationale", "").lower()
            forbidden_terms = ["f1 score", "accuracy", "auc", "roc",
                               "precision score", "recall score",
                               "false alarm rate", "cnn performance",
                               "model accuracy", "validation loss",
                               "test accuracy", "training loss"]
            for term in forbidden_terms:
                self.assertNotIn(term, rationale,
                                 f"Forbidden metric term '{term}' found in rationale")
        print("  -> PASS: No model performance metrics in decision rationale.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
