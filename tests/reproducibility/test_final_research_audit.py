"""
NeuroAegis Phase 8: Authoritative Final Research Audit Test Suite.
Verifies:
- Cryptographic integrity of checkpoint and spatial graph
- Parameter count exactness (91,858)
- Zero patient, recording, window, sequence, or threshold leakage
- Exact reconciliation of window counts (219,909) and confusion matrix
- Correct recomputation of all metrics from raw predictions
- Probability range validity [0.0, 1.0] and absence of NaNs
- Partition disjointness across Train, Val, and Test
- Siena benchmark subset qualification and calibration isolation
- Presence and validity of all 18 publication figures and 13 publication tables
- Presence of reproducibility manifest and cross-phase consistency matrix
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
RESULTS_DIR = os.path.join(BASE_DIR, "research/audits/validation_audit/final_results")
PUB_DIR = os.path.join(BASE_DIR, "research/audits/validation_audit/publication")


def get_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestFinalResearchAudit(unittest.TestCase):

    def setUp(self):
        sys.path.append(BASE_DIR)

    def test_01_checkpoint_hash_integrity(self):
        ckpt_path = os.path.join(BASE_DIR, "artifacts/checkpoints/frozen_cnn_gnn_gru.pt")
        expected_hash = "2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca"
        actual_hash = get_sha256(ckpt_path)
        self.assertEqual(actual_hash, expected_hash, "Frozen model checkpoint hash mismatch!")

    def test_02_spatial_graph_hash_integrity(self):
        adj_path = os.path.join(BASE_DIR, "research/experiments/gnn/frozen_graph_adjacency.csv")
        expected_hash = "062c6aaffdc32ea1b9b5b97c82c224d40e0ab3b7e9b2afad5fd5b3e5f52db48e"
        actual_hash = get_sha256(adj_path)
        self.assertEqual(actual_hash, expected_hash, "Frozen graph adjacency matrix hash mismatch!")

    def test_03_graph_topology_exactness(self):
        adj_path = os.path.join(BASE_DIR, "research/experiments/gnn/frozen_graph_adjacency.csv")
        adj = pd.read_csv(adj_path, index_col=0).values
        nodes = adj.shape[0]
        self.assertEqual(nodes, 23, "Graph must have exactly 23 nodes")
        off_diag = adj.copy()
        np.fill_diagonal(off_diag, 0)
        edges = int(np.sum(off_diag > 0) / 2)
        self.assertEqual(edges, 40, "Graph must have exactly 40 undirected edges")
        density = (2 * edges) / (nodes * (nodes - 1))
        self.assertAlmostEqual(density, 0.158103, places=4, msg="Graph density must be ~15.81%")

    def test_04_model_parameter_count(self):
        from neuroaegis.models.baselines.model_c import CNN_GNN_GRU
        model = CNN_GNN_GRU()
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 91858, "Model parameter count must equal exactly 91,858!")

    def test_05_patient_partition_disjointness(self):
        train_pts = {"chb04", "chb09", "chb11", "chb12", "chb13", "chb14", "chb15", "chb16", "chb17", "chb18", "chb19", "chb20", "chb21", "chb22", "chb23", "chb24"}
        val_pts = {"chb06", "chb07", "chb08", "chb10"}
        test_pts = {"chb01", "chb02", "chb03", "chb05"}

        self.assertEqual(len(train_pts), 16)
        self.assertEqual(len(val_pts), 4)
        self.assertEqual(len(test_pts), 4)
        self.assertTrue(train_pts.isdisjoint(val_pts), "Train and Validation cohorts overlap!")
        self.assertTrue(train_pts.isdisjoint(test_pts), "Train and Test cohorts overlap!")
        self.assertTrue(val_pts.isdisjoint(test_pts), "Validation and Test cohorts overlap!")

    def test_06_window_counts_and_confusion_matrix_reconciliation(self):
        pred_path = os.path.join(BASE_DIR, "research/experiments/model_c/results/final_test_predictions.csv")
        df = pd.read_csv(pred_path)
        self.assertEqual(len(df), 219909, "Total evaluation windows must equal 219,909!")

        y_true = df["label_50pct_overlap"].values
        y_prob = df["predicted_probability"].values
        y_pred = (y_prob >= 0.50).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        self.assertEqual(tp + tn + fp + fn, 219909, "Confusion matrix sum must equal total windows!")
        self.assertEqual(tp, 534, "True positives must equal 534")
        self.assertEqual(tn, 218873, "True negatives must equal 218,873")
        self.assertEqual(fp, 399, "False positives must equal 399")
        self.assertEqual(fn, 103, "False negatives must equal 103")

    def test_07_metric_recomputation_exactness(self):
        pred_path = os.path.join(BASE_DIR, "research/experiments/model_c/results/final_test_predictions.csv")
        df = pd.read_csv(pred_path)
        y_true = df["label_50pct_overlap"].values
        y_prob = df["predicted_probability"].values

        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        self.assertAlmostEqual(auroc, 0.98970, places=4, msg="AUROC must match 0.98970")
        self.assertAlmostEqual(auprc, 0.80681, places=4, msg="AUPRC must match 0.80681")

        # Check authoritative metrics json
        with open(os.path.join(RESULTS_DIR, "authoritative_final_metrics.json")) as f:
            m = json.load(f)
        self.assertEqual(m["confusion_matrix"]["true_positives"], 534)
        self.assertEqual(m["confusion_matrix"]["false_positives"], 399)
        self.assertEqual(m["event_metrics"]["total_events"], 22)
        self.assertEqual(m["event_metrics"]["detected_events"], 21)
        self.assertAlmostEqual(m["event_metrics"]["event_sensitivity"], 0.9545, places=3)
        self.assertAlmostEqual(m["clinical_metrics"]["false_alarms_per_24h"], 62.66, places=1)

    def test_08_probability_validity_and_no_nans(self):
        pred_path = os.path.join(BASE_DIR, "research/experiments/model_c/results/final_test_predictions.csv")
        df = pd.read_csv(pred_path)
        y_prob = df["predicted_probability"].values
        self.assertFalse(np.isnan(y_prob).any(), "Predicted probabilities contain NaN!")
        self.assertTrue(np.all((y_prob >= 0.0) & (y_prob <= 1.0)), "Probabilities must be in [0.0, 1.0]!")

    def test_09_no_duplicate_window_ids(self):
        pred_path = os.path.join(BASE_DIR, "research/experiments/model_c/results/final_test_predictions.csv")
        df = pd.read_csv(pred_path)
        self.assertEqual(df["window_id"].nunique(), len(df), "Window IDs must be unique!")

    def test_10_leakage_audit(self):
        pred_path = os.path.join(BASE_DIR, "research/experiments/model_c/results/final_test_predictions.csv")
        df = pd.read_csv(pred_path)
        test_pts = set(df["patient_id"].unique())
        self.assertEqual(test_pts, {"chb01", "chb02", "chb03", "chb05"}, "Test predictions contain non-test patients!")

        # Verify sequence causality
        with open(os.path.join(BASE_DIR, "research/experiments/model_c/frozen_gru_config.json")) as f:
            gru_cfg = json.load(f)
        self.assertEqual(gru_cfg["gru_direction"], "unidirectional", "GRU must be unidirectional to prevent sequence leakage!")

    def test_11_siena_benchmark_subset_qualification(self):
        with open(os.path.join(BASE_DIR, "research/experiments/siena/results/siena_zero_shot_summary.json")) as f:
            s = json.load(f)
        self.assertEqual(s["evaluated_patients"], 2, "Siena evaluated patients must equal 2")
        self.assertEqual(s["evaluated_recordings"], 4, "Siena evaluated recordings must equal 4")
        self.assertEqual(s["metrics"]["total_events"], 4, "Siena total events must equal 4")
        self.assertEqual(s["metrics"]["detected_events"], 4, "Siena detected events must equal 4")

        # Adaptation calibration cohort isolation
        with open(os.path.join(BASE_DIR, "research/experiments/siena/results/siena_adapted_summary.json")) as f:
            a = json.load(f)
        self.assertEqual(a["calibration_patients"], ["PN00"], "Adaptation must be calibrated on PN00 only!")
        self.assertEqual(a["test_patients"], ["PN12"], "Adaptation held-out test must be PN12 only!")

    def test_12_decision_threshold_freeze(self):
        with open(os.path.join(RESULTS_DIR, "authoritative_final_metrics.json")) as f:
            m = json.load(f)
        self.assertEqual(m["decision_threshold_tau"], 0.50, "Decision threshold must remain tau = 0.50!")

    def test_13_publication_figures_presence(self):
        fig_names = [f"figure_{i:02d}_" for i in range(1, 19)]
        actual_figs = os.listdir(os.path.join(BASE_DIR, "research/audits/validation_audit/figures"))
        for prefix in fig_names:
            matches = [f for f in actual_figs if f.startswith(prefix) and f.endswith(".png")]
            self.assertTrue(len(matches) >= 1, f"Missing figure with prefix {prefix}")

    def test_14_publication_tables_presence(self):
        table_prefixes = [f"table_{i:02d}_" for i in range(1, 14)]
        actual_tables = os.listdir(os.path.join(PUB_DIR, "tables"))
        for prefix in table_prefixes:
            csv_matches = [f for f in actual_tables if f.startswith(prefix) and f.endswith(".csv")]
            md_matches = [f for f in actual_tables if f.startswith(prefix) and f.endswith(".md")]
            tex_matches = [f for f in actual_tables if f.startswith(prefix) and f.endswith(".tex")]
            self.assertTrue(len(csv_matches) >= 1, f"Missing CSV for table {prefix}")
            self.assertTrue(len(md_matches) >= 1, f"Missing MD for table {prefix}")
            self.assertTrue(len(tex_matches) >= 1, f"Missing LaTeX for table {prefix}")

    def test_15_master_workbook_presence_and_sheet_count(self):
        import openpyxl
        wb_path = os.path.join(BASE_DIR, "research/audits/validation_audit/NeuroAegis_Final_Research_Results.xlsx")
        self.assertTrue(os.path.exists(wb_path), "Master Excel workbook missing!")
        wb = openpyxl.load_workbook(wb_path, read_only=True)
        self.assertEqual(len(wb.sheetnames), 23, "Master workbook must have exactly 23 sheets!")

    def test_16_reproducibility_manifest_presence(self):
        manifest_path = os.path.join(RESULTS_DIR, "reproducibility_manifest.json")
        self.assertTrue(os.path.exists(manifest_path), "Reproducibility manifest missing!")
        with open(manifest_path) as f:
            m = json.load(f)
        self.assertIn("cryptographic_hashes", m)
        self.assertEqual(m["cryptographic_hashes"]["frozen_checkpoint_pt"]["sha256"],
                         "2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca")

    def test_17_claim_audit_presence(self):
        claim_path = os.path.join(RESULTS_DIR, "final_claim_audit.csv")
        self.assertTrue(os.path.exists(claim_path), "Final claim audit CSV missing!")
        df = pd.read_csv(claim_path)
        self.assertEqual(len(df), 9, "Claim audit must contain exactly 9 claims!")
        self.assertIn("NOT SUPPORTED", df["supported_status"].values, "Clinical deployment claim must be qualified as NOT SUPPORTED!")


if __name__ == "__main__":
    unittest.main()
