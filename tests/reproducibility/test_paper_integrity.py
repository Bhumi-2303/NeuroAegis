import unittest
import os
import hashlib
import pandas as pd
import openpyxl

class TestPhase9PaperIntegrity(unittest.TestCase):
    def setUp(self):
        self.base_dir = "research/experiments/temporal_post_processing"
        self.manuscript_path = os.path.join(self.base_dir, "manuscript/NeuroAegis_Final_Manuscript.md")
        with open(self.manuscript_path, "r") as f:
            self.manuscript_text = f.read()

    def test_file_structure_exists(self):
        required_files = [
            "manuscript/NeuroAegis_Final_Manuscript.md",
            "tables/table_01_dataset.csv",
            "tables/table_02_partition.csv",
            "tables/table_03_architecture.csv",
            "tables/table_04_performance.csv",
            "tables/table_05_ablation.csv",
            "tables/table_06_patient_level.csv",
            "tables/table_07_event_level.csv",
            "tables/table_08_cross_domain.csv",
            "tables/table_09_adaptation.csv",
            "tables/table_10_xai.csv",
            "tables/table_11_statistics.csv",
            "tables/table_12_computation.csv",
            "evidence/evidence_map.xlsx",
            "evidence/evidence_map.csv",
            "audit/paper_audit.md",
            "audit/numerical_claim_audit.csv",
            "audit/citation_audit.csv",
            "supplementary/supplementary_results.md",
            "supplementary/reproducibility.md",
            "final/abstract.md",
            "final/contributions.md",
            "final/limitations.md",
            "final/conclusion.md",
        ]
        for rel_path in required_files:
            full_path = os.path.join(self.base_dir, rel_path)
            self.assertTrue(os.path.exists(full_path), f"Missing required Phase 9 file: {full_path}")
            self.assertGreater(os.path.getsize(full_path), 0, f"File is empty: {full_path}")

    def test_authoritative_metrics_in_manuscript(self):
        key_strings = [
            "83.83%",      # Window Sensitivity
            "99.82%",      # Window Specificity
            "57.24%",      # Precision
            "0.68025",     # F1 Score
            "0.98970",     # AUROC
            "0.80681",     # AUPRC
            "95.45%",      # Event Sensitivity
            "62.66",       # False alarms / 24h
            "9.0 seconds", # Median detection delay
            "91,858",      # Total parameters
            "52,497",      # Backbone parameters
            "39,361",      # Trainable GRU parameters
            "40",          # Spatial graph edges
            "23",          # Spatial graph nodes
            "0.30",        # Graph threshold theta
            "0.50",        # Decision threshold tau
            "219,909",     # Total evaluation windows
            "152.82",      # Continuous monitoring duration hours
            "5.0",         # Window length (seconds)
            "2.5",         # Window stride (seconds)
            "22.5",        # Temporal context (seconds)
            "chb01_15",    # Missed seizure identification
        ]
        for s in key_strings:
            self.assertIn(s, self.manuscript_text, f"Key metric or string '{s}' not found in manuscript.")

    def test_evidence_map_completeness(self):
        csv_path = os.path.join(self.base_dir, "evidence/evidence_map.csv")
        df = pd.read_csv(csv_path)
        self.assertGreaterEqual(len(df), 35, f"Evidence map has only {len(df)} entries, expected >= 35")
        self.assertTrue((df['Verified'] == 'PASS').all(), "Not all evidence claims are verified as PASS")

        xlsx_path = os.path.join(self.base_dir, "evidence/evidence_map.xlsx")
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.active
        self.assertEqual(ws.title, "Evidence_Map")
        self.assertGreaterEqual(ws.max_row, 36)

    def test_citation_audit_integrity(self):
        csv_path = os.path.join(self.base_dir, "audit/citation_audit.csv")
        df = pd.read_csv(csv_path)
        self.assertGreaterEqual(len(df), 20, f"Citation audit has only {len(df)} entries, expected >= 20")
        self.assertTrue((df['Audit Status'] == 'Verified').all(), "Not all citations are verified")

    def test_checkpoint_and_adjacency_hashes(self):
        ckpt_path = "artifacts/checkpoints/frozen_cnn_gnn_gru.pt"
        adj_path = "research/experiments/gnn/frozen_graph_adjacency.csv"
        
        expected_ckpt = "2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca"
        expected_adj = "062c6aaffdc32ea1b9b5b97c82c224d40e0ab3b7e9b2afad5fd5b3e5f52db48e"
        
        h_ckpt = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
        h_adj = hashlib.sha256(open(adj_path, "rb").read()).hexdigest()
        
        self.assertEqual(h_ckpt, expected_ckpt, "Model checkpoint hash modified!")
        self.assertEqual(h_adj, expected_adj, "Graph adjacency hash modified!")

    def test_siena_subset_clear_qualification(self):
        self.assertIn("benchmark subset", self.manuscript_text)
        self.assertIn("2 patients", self.manuscript_text)
        self.assertIn("4 seizures", self.manuscript_text)

    def test_clinician_validation_correct_reporting(self):
        self.assertIn("Clinician validation was not performed", self.manuscript_text)

    def test_no_non_overlapping_mention(self):
        self.assertNotIn("non-overlapping", self.manuscript_text.lower(), "Erroneous 'non-overlapping' found in manuscript.")

if __name__ == "__main__":
    unittest.main()
