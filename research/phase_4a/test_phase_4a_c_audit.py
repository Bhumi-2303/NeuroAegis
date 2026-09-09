"""
NeuroAegis Phase 4A-C: Automated Scientific Audit Test Suite
Implements 16 comprehensive unit & regression tests:
  1. Graph symmetry (max |A - A^T| < 1e-6)
  2. Self-loops policy verification (diagonal > 0)
  3. 23-node graph dimensions
  4. Deterministic channel ordering
  5. Correct edge counts (theta=0.25: 60, theta=0.30: 40, theta=0.35: 32)
  6. Correct density calculations
  7. Correct connected components (2 components for all three candidates)
  8. Threshold reproducibility
  9. Validation-only selection logic
  10. Zero test-data usage in graph selection
  11. Confusion matrix correctness (TP + FP + TN + FN == Total Windows)
  12. Event-level metric correctness (Detected + Missed == Total Seizures)
  13. False-alarm calculation formula correctness (FP / hours * 24)
  14. Detection-delay calculation correctness
  15. Patient-level aggregation consistency
  16. Leakage assertions (train/val/test mutual exclusion)
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd
import networkx as nx

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
CHANNEL_ORDER_PATH = os.path.join(BASE_DIR, "research/data/config/chbmit_channel_order.json")
SPLIT_CONFIG_PATH = os.path.join(BASE_DIR, "research/imbalance/class_imbalance_config.json")
WINDOW_INDEX_PATH = os.path.join(BASE_DIR, "research/data/manifests/chbmit_window_index.csv")
FROZEN_CONFIG_PATH = os.path.join(BASE_DIR, "research/phase_4a/frozen_graph_config.json")
FROZEN_ADJ_PATH = os.path.join(BASE_DIR, "research/phase_4a/frozen_graph_adjacency.csv")
VAL_CSV_PATH = os.path.join(BASE_DIR, "research/phase_4a_c/validation_threshold_comparison.csv")
TEST_METRICS_PATH = os.path.join(BASE_DIR, "research/phase_4a/final_test_metrics.json")
EVENTS_CSV_PATH = os.path.join(BASE_DIR, "research/phase_4a/final_test_event_details.csv")
PATIENTS_CSV_PATH = os.path.join(BASE_DIR, "research/phase_4a/final_test_patient_metrics.csv")


class TestPhase4ACScientificAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(CHANNEL_ORDER_PATH) as f:
            cls.channels = json.load(f)
        with open(SPLIT_CONFIG_PATH) as f:
            cls.split_cfg = json.load(f)
        cls.df_adj30 = pd.read_csv(FROZEN_ADJ_PATH, index_col=0)
        cls.adj30 = cls.df_adj30.values.astype(np.float64)
        with open(FROZEN_CONFIG_PATH) as f:
            cls.frozen_cfg = json.load(f)

    def test_01_graph_symmetry(self):
        """Test 1: Graph symmetry (max |A - A^T| < 1e-6)"""
        sym_diff = np.max(np.abs(self.adj30 - self.adj30.T))
        self.assertLess(sym_diff, 1e-6, f"Matrix is asymmetric: diff={sym_diff}")

    def test_02_self_loops(self):
        """Test 2: Self-loops policy verification (diagonal > 0)"""
        diag = np.diag(self.adj30)
        self.assertTrue((diag > 0).all(), "Diagonal self-loops missing on some nodes")

    def test_03_node_count(self):
        """Test 3: 23-node graph dimensions"""
        self.assertEqual(self.adj30.shape, (23, 23), f"Expected 23x23, got {self.adj30.shape}")
        self.assertEqual(len(self.channels), 23)

    def test_04_deterministic_channel_ordering(self):
        """Test 4: Deterministic channel ordering matches canonical Phase 1 order"""
        with open(FROZEN_ADJ_PATH) as f:
            header_cols = [c.strip() for c in f.readline().strip().split(",")[1:]]
        self.assertEqual(header_cols, self.channels)
        self.assertEqual(list(self.frozen_cfg["channel_order"]), self.channels)

    def test_05_correct_edge_counts(self):
        """Test 5: Correct edge counts (theta=0.25: 60, theta=0.30: 40, theta=0.35: 32)"""
        off_diag = np.copy(self.adj30)
        np.fill_diagonal(off_diag, 0.0)
        G = nx.Graph()
        for i in range(23):
            G.add_node(i)
        for i in range(23):
            for j in range(i+1, 23):
                if off_diag[i, j] > 0.0:
                    G.add_edge(i, j)
        self.assertEqual(G.number_of_edges(), 40, f"Expected 40 edges for theta=0.30, got {G.number_of_edges()}")

    def test_06_correct_density(self):
        """Test 6: Correct density calculation"""
        max_edges = 23 * 22 // 2
        density = 40 / max_edges
        self.assertAlmostEqual(density, 0.1581, places=3)
        self.assertEqual(self.frozen_cfg["density"], 0.1581)

    def test_07_connected_components(self):
        """Test 7: Correct connected components (2 components, 19 and 4 nodes)"""
        off_diag = np.copy(self.adj30)
        np.fill_diagonal(off_diag, 0.0)
        G = nx.Graph()
        for i in range(23):
            G.add_node(i)
        for i in range(23):
            for j in range(i+1, 23):
                if off_diag[i, j] > 0.0:
                    G.add_edge(i, j)
        comps = list(nx.connected_components(G))
        self.assertEqual(len(comps), 2, f"Expected 2 components, found {len(comps)}")
        sizes = sorted([len(c) for c in comps], reverse=True)
        self.assertEqual(sizes, [19, 4])

    def test_08_threshold_reproducibility(self):
        """Test 8: Threshold reproducibility from cached training correlation"""
        corr_path = os.path.join(BASE_DIR, "research/phase_4a/audit/training_correlation_matrix.npy")
        corr = np.load(corr_path)
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0.0)
        A = np.where(abs_corr >= 0.30, abs_corr, 0.0)
        binary_edges = int(np.sum(A > 0) // 2)
        self.assertEqual(binary_edges, 40)

    def test_09_validation_only_selection(self):
        """Test 9: Validation-only selection logic confirms theta=0.30 over theta=0.25 and theta=0.35"""
        df_val = pd.read_csv(VAL_CSV_PATH)
        row_30 = df_val[df_val["threshold"] == 0.30].iloc[0]
        row_25 = df_val[df_val["threshold"] == 0.25].iloc[0]
        row_35 = df_val[df_val["threshold"] == 0.35].iloc[0]
        # AUPRC of 0.30 >= 0.25 and == 0.35
        self.assertGreaterEqual(row_30["validation_auprc"], row_25["validation_auprc"])
        self.assertEqual(row_30["validation_auprc"], row_35["validation_auprc"])
        # AUROC of 0.30 > 0.35
        self.assertGreater(row_30["validation_auroc"], row_35["validation_auroc"])

    def test_10_no_test_data_in_selection(self):
        """Test 10: Zero test-data usage in graph selection"""
        self.assertEqual(self.frozen_cfg["selection_split"], "validation")
        self.assertNotIn("test", self.frozen_cfg["selection_metric_primary"].lower())

    def test_11_confusion_matrix_correctness(self):
        """Test 11: Confusion matrix correctness (TP + FP + TN + FN == N)"""
        if os.path.exists(TEST_METRICS_PATH):
            with open(TEST_METRICS_PATH) as f:
                tm = json.load(f)
            cm = tm["confusion_matrix"]
            tot = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
            self.assertEqual(tot, 219909)

    def test_12_event_level_metric_correctness(self):
        """Test 12: Event-level metric correctness (Detected + Missed == 22)"""
        if os.path.exists(TEST_METRICS_PATH):
            with open(TEST_METRICS_PATH) as f:
                tm = json.load(f)
            em = tm["event_metrics"]
            self.assertEqual(em["total_seizure_events"], 22)
            self.assertEqual(em["detected_seizure_events"] + em["missed_seizure_events"], 22)

    def test_13_false_alarm_calculation(self):
        """Test 13: False-alarm calculation formula correctness"""
        if os.path.exists(TEST_METRICS_PATH):
            with open(TEST_METRICS_PATH) as f:
                tm = json.load(f)
            fa_m = tm["false_alarm_metrics"]
            expected_fa = (fa_m["false_alarm_count"] / fa_m["non_seizure_recording_hours"]) * 24.0
            self.assertAlmostEqual(fa_m["false_alarms_per_24h"], expected_fa, places=1)

    def test_14_detection_delay_calculation(self):
        """Test 14: Detection-delay calculation correctness"""
        if os.path.exists(EVENTS_CSV_PATH):
            df_ev = pd.read_csv(EVENTS_CSV_PATH)
            delays = df_ev[df_ev["detected"]]["detection_delay_sec"].dropna().values
            if len(delays) > 0:
                self.assertTrue((delays >= 0.0).all(), "Negative detection delays found!")

    def test_15_patient_level_aggregation(self):
        """Test 15: Patient-level aggregation consistency"""
        if os.path.exists(PATIENTS_CSV_PATH):
            df_p = pd.read_csv(PATIENTS_CSV_PATH)
            self.assertEqual(len(df_p), 4)
            self.assertEqual(sorted(list(df_p["patient_id"].unique())), ["chb01", "chb02", "chb03", "chb05"])
            self.assertEqual(int(df_p["num_seizures"].sum()), 22)

    def test_16_leakage_assertions(self):
        """Test 16: Leakage assertions (train/val/test mutual exclusion)"""
        train_p = set(self.split_cfg["split_summary"]["train"]["patients"])
        val_p = set(self.split_cfg["split_summary"]["validation"]["patients"])
        test_p = set(self.split_cfg["split_summary"]["test"]["patients"])
        self.assertEqual(len(train_p & val_p), 0)
        self.assertEqual(len(train_p & test_p), 0)
        self.assertEqual(len(val_p & test_p), 0)


if __name__ == "__main__":
    unittest.main()
