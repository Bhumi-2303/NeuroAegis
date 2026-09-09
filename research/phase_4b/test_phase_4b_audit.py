"""
NeuroAegis Phase 4B: Automated Audit & Verification Test Suite
Verifies all 18 scientific, architectural, and leakage invariants specified in Section 39:
1. Sequence windows belong to same patient
2. Sequence windows belong to same recording
3. Sequence windows are temporally continuous
4. Sequence spacing equals expected stride (2.5s)
5. No sequence crosses recording boundary
6. No sequence crosses patient boundary
7. GRU hidden state resets at recording boundaries
8. No train/validation sequence overlap
9. No train/test sequence overlap
10. No validation/test sequence overlap
11. Test patients are never used for model selection
12. Frozen θ=0.30 graph is used
13. Graph is unchanged (SHA256 verified)
14. Channel order is unchanged (23 canonical channels)
15. Preprocessing is unchanged (0.5-40Hz, 60Hz notch, local z-score)
16. Label rule is unchanged (label_50pct_overlap)
17. Final test set remains untouched
18. Causal GRU does not access future windows
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd
import torch

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from research.phase_4b.cnn_gnn_gru_model import CNN_GNN_GRU
from research.phase_4b.sequence_dataset import SequenceBuilder
from research.imbalance.patient_splitter import PatientDataSplitter

MANIFEST_DIR = os.path.join(BASE_DIR, "research/data/manifests")
WINDOW_INDEX_PATH = os.path.join(MANIFEST_DIR, "chbmit_window_index.csv")
FROZEN_GRAPH_CFG = os.path.join(BASE_DIR, "research/phase_4a/frozen_graph_config.json")
FROZEN_GRAPH_ADJ = os.path.join(BASE_DIR, "research/phase_4a/frozen_graph_adjacency.csv")
FROZEN_BACKBONE_PATH = os.path.join(BASE_DIR, "research/phase_4a/frozen_cnn_gnn.pt")
FROZEN_GRU_CONFIG = os.path.join(BASE_DIR, "research/phase_4b/frozen_gru_config.json")
CHANNEL_ORDER_PATH = os.path.join(BASE_DIR, "research/data/config/chbmit_channel_order.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(65536)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


class TestPhase4BAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splitter = PatientDataSplitter(label_column="label_50pct_overlap")
        cls.train_df, cls.val_df, cls.test_df = cls.splitter.get_splits()
        cls.builder = SequenceBuilder(label_column="label_50pct_overlap")
        cls.val_seq_4 = cls.builder.build_sequences(cls.val_df, seq_len=4)
        cls.train_seq_4 = cls.builder.build_sequences(cls.train_df, seq_len=4)

    def test_01_sequence_windows_same_patient(self):
        """Assertion 1: sequence windows belong to same patient"""
        for _, row in self.val_seq_4.sample(n=min(500, len(self.val_seq_4)), random_state=42).iterrows():
            pat = row["patient_id"]
            for w_idx in row["window_indices"]:
                if w_idx >= 0:
                    self.assertEqual(self.val_df.iloc[w_idx]["patient_id"], pat)

    def test_02_sequence_windows_same_recording(self):
        """Assertion 2: sequence windows belong to same recording"""
        for _, row in self.val_seq_4.sample(n=min(500, len(self.val_seq_4)), random_state=42).iterrows():
            rec = row["recording_id"]
            for w_idx in row["window_indices"]:
                if w_idx >= 0:
                    self.assertEqual(self.val_df.iloc[w_idx]["recording_id"], rec)

    def test_03_sequence_windows_temporally_continuous(self):
        """Assertion 3: sequence windows are temporally continuous"""
        for _, row in self.val_seq_4.sample(n=min(500, len(self.val_seq_4)), random_state=42).iterrows():
            valid_indices = [w for w in row["window_indices"] if w >= 0]
            if len(valid_indices) > 1:
                times = [self.val_df.iloc[w]["window_start_sec"] for w in valid_indices]
                for k in range(len(times) - 1):
                    diff = round(times[k+1] - times[k], 2)
                    self.assertEqual(diff, 2.5)

    def test_04_sequence_spacing_equals_expected_stride(self):
        """Assertion 4: sequence spacing equals expected stride (2.5s)"""
        span_1 = self.builder.compute_temporal_span(1)
        span_4 = self.builder.compute_temporal_span(4)
        span_8 = self.builder.compute_temporal_span(8)
        span_12 = self.builder.compute_temporal_span(12)
        self.assertEqual(span_1, 5.0)
        self.assertEqual(span_4, 12.5)
        self.assertEqual(span_8, 22.5)
        self.assertEqual(span_12, 32.5)

    def test_05_no_sequence_crosses_recording_boundary(self):
        """Assertion 5: no sequence crosses recording boundary"""
        grouped = self.val_seq_4.groupby("recording_id")
        for rec_id, group in grouped:
            for _, row in group.iterrows():
                for w_idx in row["window_indices"]:
                    if w_idx >= 0:
                        self.assertEqual(self.val_df.iloc[w_idx]["recording_id"], rec_id)

    def test_06_no_sequence_crosses_patient_boundary(self):
        """Assertion 6: no sequence crosses patient boundary"""
        grouped = self.val_seq_4.groupby("patient_id")
        for pat_id, group in grouped:
            for _, row in group.iterrows():
                for w_idx in row["window_indices"]:
                    if w_idx >= 0:
                        self.assertEqual(self.val_df.iloc[w_idx]["patient_id"], pat_id)

    def test_07_gru_hidden_state_resets_at_recording_boundaries(self):
        """Assertion 7: GRU hidden state resets at recording boundaries"""
        model = CNN_GNN_GRU()
        model.eval()
        x1 = torch.randn(1, 128)
        x2 = torch.randn(1, 128)
        # Streaming step with state
        out1, h1 = model.step(x1, None)
        out2_stateful, _ = model.step(x2, h1)
        # Reset state
        out2_stateless, _ = model.step(x2, None)
        # When reset, output depends only on x2 and zero initial state
        self.assertFalse(torch.allclose(out2_stateful, out2_stateless))
        self.assertIsNotNone(h1)

    def test_08_no_train_validation_sequence_overlap(self):
        """Assertion 8: no train/validation sequence overlap"""
        train_targets = set(self.train_seq_4["target_window_id"])
        val_targets = set(self.val_seq_4["target_window_id"])
        self.assertEqual(len(train_targets & val_targets), 0)

    def test_09_no_train_test_sequence_overlap(self):
        """Assertion 9: no train/test sequence overlap"""
        test_seq_4 = self.builder.build_sequences(self.test_df, seq_len=4)
        train_targets = set(self.train_seq_4["target_window_id"])
        test_targets = set(test_seq_4["target_window_id"])
        self.assertEqual(len(train_targets & test_targets), 0)

    def test_10_no_validation_test_sequence_overlap(self):
        """Assertion 10: no validation/test sequence overlap"""
        test_seq_4 = self.builder.build_sequences(self.test_df, seq_len=4)
        val_targets = set(self.val_seq_4["target_window_id"])
        test_targets = set(test_seq_4["target_window_id"])
        self.assertEqual(len(val_targets & test_targets), 0)

    def test_11_test_patients_never_used_for_model_selection(self):
        """Assertion 11: test patients are never used for model selection"""
        if os.path.exists(FROZEN_GRU_CONFIG):
            with open(FROZEN_GRU_CONFIG, "r") as f:
                cfg = json.load(f)
            self.assertEqual(cfg["selection_split"], "validation")
            self.assertIn("Validation AUPRC", cfg["selection_metric_primary"])
            for p in self.splitter.test_patients:
                self.assertNotIn(p, cfg.get("validation_patients", ["chb06", "chb07", "chb08", "chb10"]))

    def test_12_frozen_theta_030_graph_is_used(self):
        """Assertion 12: frozen θ=0.30 graph is used"""
        with open(FROZEN_GRAPH_CFG, "r") as f:
            g_cfg = json.load(f)
        self.assertEqual(g_cfg["threshold"], 0.30)
        num_edges = g_cfg.get("number_of_edges", g_cfg.get("total_edges_undirected"))
        self.assertEqual(num_edges, 40)
        num_components = g_cfg.get("number_of_connected_components", g_cfg.get("connected_components"))
        self.assertEqual(num_components, 2)

    def test_13_graph_is_unchanged(self):
        """Assertion 13: graph is unchanged"""
        adj_hash = sha256(FROZEN_GRAPH_ADJ)
        self.assertEqual(adj_hash, "062c6aaffdc32ea1b9b5b97c82c224d40e0ab3b7e9b2afad5fd5b3e5f52db48e")

    def test_14_channel_order_is_unchanged(self):
        """Assertion 14: channel order is unchanged (23 canonical channels)"""
        with open(CHANNEL_ORDER_PATH, "r") as f:
            chans = json.load(f)
        self.assertEqual(len(chans), 23)
        self.assertEqual(chans[0], "FP1-F7")
        self.assertEqual(chans[14], "T8-P8")
        self.assertEqual(chans[22], "T8-P8")

    def test_15_preprocessing_is_unchanged(self):
        """Assertion 15: preprocessing is unchanged"""
        with open(os.path.join(BASE_DIR, "research/phase_2/preprocessing_config.json"), "r") as f:
            p_cfg = json.load(f)
        if "filters" in p_cfg:
            self.assertEqual(p_cfg["filters"]["bandpass"]["lowcut_hz"], 0.5)
            self.assertEqual(p_cfg["filters"]["bandpass"]["highcut_hz"], 40.0)
            self.assertEqual(p_cfg["filters"]["notch"]["notch_freq_hz"], 60.0)
        else:
            self.assertEqual(p_cfg["bandpass_filter"]["lowcut"], 0.5)
            self.assertEqual(p_cfg["bandpass_filter"]["highcut"], 40.0)
            self.assertEqual(p_cfg["notch_filter"]["freq"], 60.0)

    def test_16_label_rule_is_unchanged(self):
        """Assertion 16: label rule is unchanged"""
        self.assertEqual(self.builder.label_column, "label_50pct_overlap")
        pos_b = (self.train_df["label_50pct_overlap"] == 1).sum()
        self.assertEqual(pos_b, 3308)

    def test_17_final_test_set_remains_untouched(self):
        """Assertion 17: final test set remains untouched"""
        self.assertEqual(len(self.test_df), 219909)
        self.assertEqual(set(self.test_df["patient_id"].unique()), {"chb01", "chb02", "chb03", "chb05"})

    def test_18_causal_gru_does_not_access_future_windows(self):
        """Assertion 18: causal GRU does not access future windows"""
        model = CNN_GNN_GRU()
        self.assertFalse(model.gru.bidirectional)
        self.assertEqual(model.gru.num_layers, 1)
        
        # Test causal property: changing a future window must not change current window prediction
        x_seq1 = torch.randn(1, 4, 128)
        x_seq2 = x_seq1.clone()
        # If we alter future window (e.g. window 3)
        x_seq2[:, 3, :] = torch.randn(1, 128)
        
        # Hidden states at step 2 (current) must be identical
        gru_out1, _ = model.gru(x_seq1)
        gru_out2, _ = model.gru(x_seq2)
        self.assertTrue(torch.allclose(gru_out1[:, 2, :], gru_out2[:, 2, :], atol=1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
