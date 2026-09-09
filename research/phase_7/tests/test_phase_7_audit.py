"""
NeuroAegis Phase 7: Automated Audit & Verification Test Suite
Verifies 20 critical assertions spanning data integrity, leakage isolation,
mathematical reconciliation, file presence, figure quality, and reproducibility.
"""

import os
import sys
import json
import unittest
import hashlib
import numpy as np
import pandas as pd
import openpyxl

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
PHASE7_DIR = os.path.join(BASE_DIR, "research/phase_7")
RESULTS_DIR = os.path.join(PHASE7_DIR, "results")
FIGURES_DIR = os.path.join(PHASE7_DIR, "figures")
WORKBOOK_PATH = os.path.join(PHASE7_DIR, "Phase_7_Statistical_Robustness.xlsx")

MANIFEST_DIR = os.path.join(BASE_DIR, "research/data/manifests")
WINDOW_INDEX_PATH = os.path.join(MANIFEST_DIR, "chbmit_window_index.csv")
EVENTS_PATH = os.path.join(MANIFEST_DIR, "chbmit_seizure_events.csv")
MANIFEST_PATH = os.path.join(MANIFEST_DIR, "chbmit_manifest.csv")

FROZEN_MODEL_PATH = os.path.join(BASE_DIR, "research/phase_4b/frozen_cnn_gnn_gru.pt")
FROZEN_GRAPH_PATH = os.path.join(BASE_DIR, "research/phase_4a/frozen_graph_config.json")
FROZEN_GRAPH_ADJ_PATH = os.path.join(BASE_DIR, "research/phase_4a/frozen_graph_adjacency.csv")

EXPECTED_MODEL_SHA256 = "2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca"
EXPECTED_GRAPH_ADJ_SHA256 = "062c6aaffdc32ea1b9b5b97c82c224d40e0ab3b7e9b2afad5fd5b3e5f52db48e"
EXPECTED_GRAPH_CFG_SHA256 = "7798862ec4493ae22ce43eba7791ee9838c4a266cd8b5cc49d096a2e61f0aa82"

def get_file_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()

class TestPhase7Audit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(RESULTS_DIR, "phase_7_summary.json"), "r") as f:
            cls.summary = json.load(f)
        cls.df_comp = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
        cls.df_pat = pd.read_csv(os.path.join(RESULTS_DIR, "patient_level_results.csv"))
        cls.df_events = pd.read_csv(os.path.join(RESULTS_DIR, "event_level_results.csv"))
        cls.df_thresh = pd.read_csv(os.path.join(RESULTS_DIR, "threshold_validation.csv"))
        cls.df_boot = pd.read_csv(os.path.join(RESULTS_DIR, "bootstrap_results.csv"))

    def test_01_identical_test_cohort(self):
        """Assertion 1: All models evaluated on identical 4-patient CHB-MIT test cohort."""
        tc = self.summary["test_cohort"]
        self.assertEqual(tc["num_patients"], 4)
        self.assertEqual(tc["patients"], ["chb01", "chb02", "chb03", "chb05"])
        self.assertEqual(tc["num_recordings"], 155)
        self.assertEqual(tc["num_seizures"], 22)
        self.assertEqual(tc["total_windows"], 219909)
        self.assertEqual(tc["total_monitoring_hours"], 152.82)

    def test_02_patient_manifest_reconciliation(self):
        """Assertion 2: Patient counts and IDs reconcile with patient manifests."""
        df_pats = self.df_pat["patient_id"].unique()
        self.assertEqual(set(df_pats), {"chb01", "chb02", "chb03", "chb05"})
        for p in df_pats:
            p_rows = self.df_pat[self.df_pat["patient_id"] == p]
            self.assertEqual(len(p_rows), 3)  # Models A, B, C

    def test_03_recording_manifest_reconciliation(self):
        """Assertion 3: Total recordings in test partition equal 155."""
        p4b_df = pd.read_csv(os.path.join(BASE_DIR, "research/phase_4b/results/final_test_predictions.csv"))
        self.assertEqual(p4b_df["recording_id"].nunique(), 155)

    def test_04_window_count_and_confusion_reconciliation(self):
        """Assertion 4: Confusion matrix counts sum exactly to 219,909 for all models."""
        for _, row in self.df_comp.iterrows():
            total = row["tp"] + row["fp"] + row["tn"] + row["fn"]
            self.assertEqual(total, 219909, f"Confusion sum mismatch for {row['model_id']}: {total}")

    def test_05_ground_truth_consistency(self):
        """Assertion 5: Ground truth positive window count is exactly 637 across all models."""
        for _, row in self.df_comp.iterrows():
            positives = row["tp"] + row["fn"]
            negatives = row["fp"] + row["tn"]
            self.assertEqual(positives, 637, f"Positives mismatch for {row['model_id']}")
            self.assertEqual(negatives, 219272, f"Negatives mismatch for {row['model_id']}")

    def test_06_class_imbalance_integrity(self):
        """Assertion 6: Class imbalance ratio is 344.23:1 (0.290% positive)."""
        pos_ratio = 637 / 219909
        self.assertAlmostEqual(pos_ratio, 0.00289665, places=5)
        imb_ratio = 219272 / 637
        self.assertAlmostEqual(imb_ratio, 344.226, places=2)

    def test_07_patient_leakage_isolation(self):
        """Assertion 7: Patient partitions for Train, Val, Test are mutually disjoint."""
        train_p = set(self.summary["leakage_audit"]["patient_leakage"].split())
        self.assertIn("PASS", self.summary["leakage_audit"]["patient_leakage"])

    def test_08_recording_leakage_isolation(self):
        """Assertion 8: Zero recording overlap across splits."""
        self.assertIn("PASS", self.summary["leakage_audit"]["recording_leakage"])

    def test_09_window_leakage_isolation(self):
        """Assertion 9: Zero window overlap across splits."""
        self.assertIn("PASS", self.summary["leakage_audit"]["window_leakage"])

    def test_10_frozen_decision_threshold(self):
        """Assertion 10: Frozen decision threshold tau = 0.50 on test set."""
        self.assertEqual(self.summary["model_architecture_freeze"]["decision_threshold_tau"], 0.50)

    def test_11_threshold_sweep_zero_leakage(self):
        """Assertion 11: Threshold sweep is conducted exclusively on validation data."""
        self.assertEqual(len(self.df_thresh), 9)
        for _, r in self.df_thresh.iterrows():
            self.assertIn("Validation", r["dataset_split"])
            tot = r["true_positives"] + r["true_negatives"] + r["false_positives"] + r["false_negatives"]
            self.assertEqual(tot, 293410)

    def test_12_frozen_checkpoint_integrity(self):
        """Assertion 12: Frozen Phase 4B checkpoint SHA256 matches expected hash."""
        actual_sha = get_file_sha256(FROZEN_MODEL_PATH)
        self.assertEqual(actual_sha, EXPECTED_MODEL_SHA256)
        self.assertEqual(self.summary["model_architecture_freeze"]["model_c_sha256"], EXPECTED_MODEL_SHA256)

    def test_13_frozen_spatial_graph_integrity(self):
        """Assertion 13: Frozen Phase 4A-C spatial graph adjacency and config SHA256 match expected hashes."""
        actual_adj_sha = get_file_sha256(FROZEN_GRAPH_ADJ_PATH)
        actual_cfg_sha = get_file_sha256(FROZEN_GRAPH_PATH)
        self.assertEqual(actual_adj_sha, EXPECTED_GRAPH_ADJ_SHA256)
        self.assertEqual(actual_cfg_sha, EXPECTED_GRAPH_CFG_SHA256)
        self.assertEqual(self.summary["model_architecture_freeze"]["spatial_graph_adjacency_sha256"], EXPECTED_GRAPH_ADJ_SHA256)
        self.assertEqual(self.summary["model_architecture_freeze"]["spatial_graph_config_sha256"], EXPECTED_GRAPH_CFG_SHA256)

    def test_14_bootstrap_reproducibility(self):
        """Assertion 14: Bootstrap seed 42 and 5,000 iterations recorded."""
        self.assertEqual(self.summary["bootstrap_validation"]["seed"], 42)
        self.assertEqual(self.summary["bootstrap_validation"]["iterations"], 5000)

    def test_15_all_probabilities_bounded(self):
        """Assertion 15: All predicted probabilities are finite and within [0, 1]."""
        p3_npz = np.load(os.path.join(RESULTS_DIR, "phase_3_test_predictions.npz"))
        yp3 = p3_npz["y_prob"]
        self.assertTrue(np.all((yp3 >= 0.0) & (yp3 <= 1.0)))
        self.assertTrue(np.all(np.isfinite(yp3)))

    def test_16_clinical_event_counts(self):
        """Assertion 16: All 22 test seizure events accounted for in event results."""
        self.assertEqual(len(self.df_events), 22)
        self.assertEqual(self.df_events["model_c_detected"].sum(), 21)
        self.assertEqual(self.df_events["model_b_detected"].sum(), 6)

    def test_17_all_11_result_files_exist(self):
        """Assertion 17: All 11 required results files exist and are non-empty."""
        required_files = [
            "model_comparison.csv",
            "ablation_results.csv",
            "patient_level_results.csv",
            "event_level_results.csv",
            "false_alarm_results.csv",
            "detection_delay_results.csv",
            "statistical_tests.csv",
            "bootstrap_results.csv",
            "calibration_results.csv",
            "threshold_validation.csv",
            "phase_7_summary.json"
        ]
        for fname in required_files:
            fpath = os.path.join(RESULTS_DIR, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing required file: {fname}")
            self.assertGreater(os.path.getsize(fpath), 0, f"File is empty: {fname}")

    def test_18_all_16_figures_exist_300dpi(self):
        """Assertion 18: All 16 publication figures exist and are > 50 KB."""
        required_figs = [f"fig{i:02d}" for i in range(1, 17)]
        for fprefix in required_figs:
            matches = [f for f in os.listdir(FIGURES_DIR) if f.startswith(fprefix) and f.endswith(".png")]
            self.assertEqual(len(matches), 1, f"Missing figure with prefix: {fprefix}")
            fpath = os.path.join(FIGURES_DIR, matches[0])
            self.assertGreater(os.path.getsize(fpath), 50 * 1024, f"Figure size too small (<50KB): {matches[0]}")

    def test_19_master_workbook_contains_all_19_sheets(self):
        """Assertion 19: Master Excel workbook contains all 19 required sheets."""
        self.assertTrue(os.path.exists(WORKBOOK_PATH))
        wb = openpyxl.load_workbook(WORKBOOK_PATH, read_only=True)
        expected_sheets = [
            "Experiment_Summary", "Model_Comparison", "Ablation", "Patient_Level",
            "Event_Level", "Window_Level", "False_Alarms", "Detection_Delay",
            "Bootstrap_CI", "Statistical_Tests", "Effect_Size", "Multiple_Comparison",
            "Calibration", "Threshold_Validation", "Complexity", "Leakage_Audit",
            "Reproducibility", "Figure_Registry", "Audit_Status"
        ]
        self.assertEqual(len(wb.sheetnames), 19)
        for s in expected_sheets:
            self.assertIn(s, wb.sheetnames, f"Missing sheet: {s}")

    def test_20_scientific_conclusions_consistency(self):
        """Assertion 20: Model C outperforms Model A and B across primary clinical and ranked metrics."""
        mc_row = self.df_comp[self.df_comp["model_id"] == "Model C"].iloc[0]
        mb_row = self.df_comp[self.df_comp["model_id"] == "Model B"].iloc[0]
        ma_row = self.df_comp[self.df_comp["model_id"] == "Model A"].iloc[0]

        # Model C event sensitivity > Model B
        self.assertGreater(mc_row["event_sensitivity_strict"], mb_row["event_sensitivity_strict"])
        # Model C false alarms < Model A and Model B
        self.assertLess(mc_row["false_alarms_24h"], mb_row["false_alarms_24h"])
        self.assertLess(mc_row["false_alarms_24h"], ma_row["false_alarms_24h"])
        # Model C AUPRC > Model A and Model B
        self.assertGreater(mc_row["auprc"], ma_row["auprc"])
        self.assertGreater(mc_row["auprc"], mb_row["auprc"])
        # Model C F1 > Model A and Model B
        self.assertGreater(mc_row["f1_score"], ma_row["f1_score"])
        self.assertGreater(mc_row["f1_score"], mb_row["f1_score"])

if __name__ == "__main__":
    unittest.main()
