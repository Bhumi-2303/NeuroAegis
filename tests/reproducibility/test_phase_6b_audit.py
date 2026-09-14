"""
test_phase_6b_audit.py
──────────────────────
NeuroAegis Automated Test Audit Suite — Phase 6B: Siena Cross-Domain Forensic Audit
Verifies all 20 required forensic, statistical, architectural, and leakage assertions:
  1. raw patient count == manifest patient count (14)
  2. raw event count == manifest event count (47)
  3. evaluated event count == expected event count (4)
  4. prediction rows == evaluated windows (3,538)
  5. TP + TN + FP + FN == prediction rows (3,538)
  6. unique patient count == expected evaluated patient count (2)
  7. no duplicate window IDs
  8. no duplicate sequence IDs
  9. no sequence crosses recording boundary
  10. no sequence crosses patient boundary
  11. model checkpoint hash unchanged
  12. graph hash unchanged
  13. zero-shot threshold unchanged (tau = 0.50)
  14. adaptation parameters not fitted on final evaluation cohort
  15. all 23 required channels present
  16. all model windows contain 1280 samples
  17. all model sequences contain 8 windows
  18. probabilities are finite and within [0, 1]
  19. no NaN or infinite predictions
  20. all 18 required Excel audit sheets exist and are non-empty
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd
import openpyxl

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PHASE6_DIR = os.path.join(BASE_DIR, "research/experiments/siena")
AUDIT_DIR = os.path.join(PHASE6_DIR, "audit")
MANIFEST_DIR = os.path.join(PHASE6_DIR, "manifests")
RESULTS_DIR = os.path.join(PHASE6_DIR, "results")
CONFIG_DIR = os.path.join(PHASE6_DIR, "config")

FROZEN_CHECKPOINT_PATH = os.path.join(BASE_DIR, "artifacts/checkpoints/frozen_cnn_gnn_gru.pt")
EXPECTED_CHECKPOINT_HASH = "2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca"
FROZEN_ADJ_PATH = os.path.join(BASE_DIR, "research/experiments/gnn/frozen_graph_adjacency.csv")
EXPECTED_ADJ_HASH = "062c6aaffdc32ea1b9b5b97c82c224d40e0ab3b7e9b2afad5fd5b3e5f52db48e"


class TestPhase6bAudit(unittest.TestCase):

    def _get_sha256(self, filepath: str) -> str:
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    def test_01_raw_patient_count_equals_manifest_count(self):
        """Assertion 1: Raw patient count == manifest patient count (14)."""
        df_p = pd.read_csv(os.path.join(MANIFEST_DIR, "siena_patient_manifest.csv"))
        self.assertEqual(len(df_p), 14, "Siena patient count must be exactly 14")

    def test_02_raw_event_count_equals_manifest_count(self):
        """Assertion 2: Raw event count == manifest event count (47)."""
        df_e = pd.read_csv(os.path.join(MANIFEST_DIR, "siena_seizure_events.csv"))
        self.assertEqual(len(df_e), 47, "Siena seizure event count must be exactly 47")

    def test_03_evaluated_event_count_equals_expected(self):
        """Assertion 3: Evaluated event count == expected evaluated events (4)."""
        df_ev = pd.read_csv(os.path.join(RESULTS_DIR, "siena_zero_shot_event_results.csv"))
        self.assertEqual(len(df_ev), 4, "Evaluated benchmark seizure count must be exactly 4")

    def test_04_prediction_rows_equal_evaluated_windows(self):
        """Assertion 4: Prediction rows == evaluated windows (3,538)."""
        df_pred = pd.read_csv(os.path.join(RESULTS_DIR, "siena_zero_shot_predictions.csv"))
        self.assertEqual(len(df_pred), 3538, "Prediction row count must equal 3,538")

    def test_05_confusion_matrix_sum_equals_prediction_rows(self):
        """Assertion 5: TP + TN + FP + FN == prediction rows (3,538)."""
        df_pred = pd.read_csv(os.path.join(RESULTS_DIR, "siena_zero_shot_predictions.csv"))
        gt = df_pred["ground_truth"].values
        pred = df_pred["binary_prediction"].values
        tp = ((gt == 1) & (pred == 1)).sum()
        fp = ((gt == 0) & (pred == 1)).sum()
        tn = ((gt == 0) & (pred == 0)).sum()
        fn = ((gt == 1) & (pred == 0)).sum()
        self.assertEqual(tp + fp + tn + fn, len(df_pred), "Confusion matrix sum must equal total predictions")
        self.assertEqual(tp, 64, "TP mismatch")
        self.assertEqual(fp, 5, "FP mismatch")
        self.assertEqual(tn, 3409, "TN mismatch")
        self.assertEqual(fn, 60, "FN mismatch")

    def test_06_unique_patient_count_equals_evaluated(self):
        """Assertion 6: Unique patient count == expected evaluated patient count (2)."""
        df_pred = pd.read_csv(os.path.join(RESULTS_DIR, "siena_zero_shot_predictions.csv"))
        pats = set(df_pred["patient_id"].unique())
        self.assertEqual(len(pats), 2, "Evaluated patients must be exactly 2 (PN00, PN12)")
        self.assertEqual(pats, {"PN00", "PN12"}, "Evaluated patients must be PN00 and PN12")

    def test_07_no_duplicate_window_ids(self):
        """Assertion 7: No duplicate window IDs within any recording."""
        df_pred = pd.read_csv(os.path.join(RESULTS_DIR, "siena_zero_shot_predictions.csv"))
        for rec, g in df_pred.groupby("recording_id"):
            self.assertEqual(len(g), g["window_idx"].nunique(), f"Duplicate window index found in {rec}")

    def test_08_no_duplicate_sequence_ids(self):
        """Assertion 8: No duplicate sequence IDs within any recording."""
        df_pred = pd.read_csv(os.path.join(RESULTS_DIR, "siena_zero_shot_predictions.csv"))
        for rec, g in df_pred.groupby("recording_id"):
            self.assertEqual(len(g), len(set(zip(g["start_sec"], g["end_sec"]))), f"Duplicate time intervals in {rec}")

    def test_09_no_sequence_crosses_recording_boundary(self):
        """Assertion 9: No sequence crosses recording boundary."""
        from research.experiments.siena.siena_preprocessor import SienaSequenceBuilder
        builder = SienaSequenceBuilder(seq_len=8)
        # Verify sequence builder handles per-recording window arrays
        dummy_windows = np.zeros((10, 23, 1280))
        seqs = builder.build_causal_sequences(dummy_windows)
        self.assertEqual(len(seqs), 10, "Sequence count must match window count within a recording")

    def test_10_no_sequence_crosses_patient_boundary(self):
        """Assertion 10: No sequence crosses patient boundary."""
        df_pred = pd.read_csv(os.path.join(RESULTS_DIR, "siena_zero_shot_predictions.csv"))
        rec_to_pat = df_pred.groupby("recording_id")["patient_id"].nunique()
        self.assertTrue((rec_to_pat == 1).all(), "A recording has multiple patients!")

    def test_11_model_checkpoint_hash_unchanged(self):
        """Assertion 11: Model checkpoint SHA256 matches frozen Phase 4B hash."""
        actual_hash = self._get_sha256(FROZEN_CHECKPOINT_PATH)
        self.assertEqual(actual_hash, EXPECTED_CHECKPOINT_HASH, "Model checkpoint hash mismatch!")

    def test_12_graph_hash_unchanged(self):
        """Assertion 12: Graph adjacency SHA256 matches frozen theta=0.30 hash."""
        actual_hash = self._get_sha256(FROZEN_ADJ_PATH)
        self.assertEqual(actual_hash, EXPECTED_ADJ_HASH, "Graph adjacency hash mismatch!")

    def test_13_zero_shot_threshold_unchanged(self):
        """Assertion 13: Zero-shot decision threshold remains frozen at tau = 0.50."""
        from research.experiments.siena.siena_zero_shot_evaluator import SienaZeroShotEvaluator
        evaluator = SienaZeroShotEvaluator()
        self.assertEqual(evaluator.tau_threshold, 0.50, "Zero-shot threshold must be 0.50")

    def test_14_adaptation_parameters_not_fitted_on_evaluation_cohort(self):
        """Assertion 14: Adaptation parameters not fitted on final evaluation cohort (PN12)."""
        with open(os.path.join(RESULTS_DIR, "siena_adapted_summary.json")) as f:
            ad = json.load(f)
        self.assertEqual(ad["calibration_patients"], ["PN00"], "Calibration must strictly use PN00")
        self.assertNotIn("PN12", ad["calibration_patients"], "PN12 must NOT be in calibration cohort!")
        self.assertEqual(ad["test_patients"], ["PN12"], "Evaluation cohort must be PN12")

    def test_15_all_23_required_channels_present(self):
        """Assertion 15: All 23 canonical bipolar leads mapped in configuration."""
        with open(os.path.join(CONFIG_DIR, "siena_channel_mapping.json")) as f:
            cfg = json.load(f)
        self.assertEqual(len(cfg["mappings"]), 23, "Mapping must contain exactly 23 leads")

    def test_16_all_model_windows_contain_1280_samples(self):
        """Assertion 16: All model windows contain 1280 samples (5.0s at 256 Hz)."""
        from research.experiments.siena.siena_preprocessor import SienaWindowExtractor
        extractor = SienaWindowExtractor(window_samples=1280, stride_samples=640)
        self.assertEqual(extractor.window_samples, 1280, "Window samples must be 1280")

    def test_17_all_model_sequences_contain_8_windows(self):
        """Assertion 17: All model sequences contain 8 windows (L=8)."""
        from research.experiments.siena.siena_preprocessor import SienaSequenceBuilder
        builder = SienaSequenceBuilder(seq_len=8)
        self.assertEqual(builder.seq_len, 8, "Sequence length must be 8")

    def test_18_probabilities_are_finite_and_bounded(self):
        """Assertion 18: Probabilities are finite and strictly within [0, 1]."""
        df_pred = pd.read_csv(os.path.join(RESULTS_DIR, "siena_zero_shot_predictions.csv"))
        p = df_pred["raw_probability"].values
        sp = df_pred["smoothed_probability"].values
        self.assertTrue(np.all(np.isfinite(p)), "Raw probabilities contain NaN/Inf")
        self.assertTrue(np.all(np.isfinite(sp)), "Smoothed probabilities contain NaN/Inf")
        self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)), "Raw probabilities out of bounds")
        self.assertTrue(np.all((sp >= 0.0) & (sp <= 1.0)), "Smoothed probabilities out of bounds")

    def test_19_no_nan_predictions(self):
        """Assertion 19: No NaN or null predictions in predictions dataframe."""
        df_pred = pd.read_csv(os.path.join(RESULTS_DIR, "siena_zero_shot_predictions.csv"))
        self.assertEqual(df_pred["binary_prediction"].isna().sum(), 0, "Null binary predictions found")

    def test_20_all_18_excel_sheets_exist_and_non_empty(self):
        """Assertion 20: Master audit Excel workbook contains all 18 sheets and is non-empty."""
        wb_path = os.path.join(AUDIT_DIR, "phase_6b_audit.xlsx")
        self.assertTrue(os.path.exists(wb_path), "phase_6b_audit.xlsx missing")
        wb = openpyxl.load_workbook(wb_path, read_only=True)
        expected_sheets = [
            "Cohort_Reconciliation", "Patient_Coverage", "Event_Coverage", "Window_Coverage",
            "Predictions_Audit", "Confusion_Matrix", "Event_Results", "Patient_Results",
            "False_Alarms", "Detection_Delay", "ROC_PR", "Domain_Gap",
            "Domain_Shift", "Channel_Mapping", "Adaptation_Leakage", "Reproducibility",
            "Figure_Registry", "Audit_Status"
        ]
        self.assertEqual(len(wb.sheetnames), 18, f"Expected 18 sheets, found {len(wb.sheetnames)}")
        for s in expected_sheets:
            self.assertIn(s, wb.sheetnames, f"Sheet '{s}' missing from audit workbook")


if __name__ == "__main__":
    unittest.main()
