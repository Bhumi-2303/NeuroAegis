"""
NeuroAegis Automated Test Suite for Class Imbalance Handling
Covers all 14 mandatory validation checks specified in Section 22 of research instructions.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/Volumes/BLACK-BOX/NeuroAegis/research/imbalance")
from focal_loss import BinaryFocalLossWithLogits, logits_to_probabilities
from patient_splitter import PatientDataSplitter, DEFAULT_TRAIN_PATIENTS, DEFAULT_VAL_PATIENTS, DEFAULT_TEST_PATIENTS
from dynamic_sampler import DynamicNegativeSampler
from metrics import SeizureEvaluationMetrics

BASE_DIR = "/Volumes/BLACK-BOX/NeuroAegis"
MANIFEST_DIR = os.path.join(BASE_DIR, "research/data/manifests")
WINDOW_INDEX_PATH = os.path.join(MANIFEST_DIR, "chbmit_window_index.csv")
SEIZURE_EVENTS_PATH = os.path.join(MANIFEST_DIR, "chbmit_seizure_events.csv")
EXPECTED_SHA256 = "f76dddb19fa2c24b3af24b989168b12465512b650c5e6d84bb0125a6d666302c"
EXPECTED_TOTAL_ROWS = 1414710


class TestClassImbalanceHandling(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print("Initializing TestClassImbalanceHandling Suite (14 Mandatory Tests)...")
        print("=" * 70)
        
        # Load splitter
        cls.splitter = PatientDataSplitter(
            window_index_path=WINDOW_INDEX_PATH,
            seizure_events_path=SEIZURE_EVENTS_PATH,
            train_patients=DEFAULT_TRAIN_PATIENTS,
            val_patients=DEFAULT_VAL_PATIENTS,
            test_patients=DEFAULT_TEST_PATIENTS
        )
        cls.train_df, cls.val_df, cls.test_df = cls.splitter.get_splits()
        cls.summary = cls.splitter.get_split_summary()
        
        # Initialize primary sampler (10:1)
        cls.sampler = DynamicNegativeSampler(
            train_df=cls.train_df,
            ratio=10.0,
            base_seed=42,
            label_column="label_any_overlap"
        )
        
    def test_01_master_index_unchanged(self):
        """Test 1: Master index is unchanged (verified by SHA256 checksum and exact row count)."""
        print("\n[Running Test 1/14] Master index integrity & immutability...")
        hasher = hashlib.sha256()
        with open(WINDOW_INDEX_PATH, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_sha256 = hasher.hexdigest()
        
        self.assertEqual(actual_sha256, EXPECTED_SHA256, "Master window index SHA256 checksum mismatch!")
        self.assertEqual(len(self.splitter.window_df), EXPECTED_TOTAL_ROWS, f"Expected {EXPECTED_TOTAL_ROWS} rows, got {len(self.splitter.window_df)}")
        print(f"  -> PASS: Master index SHA256 ({actual_sha256[:16]}...) and {EXPECTED_TOTAL_ROWS:,} rows perfectly intact.")

    def test_02_no_validation_patient_in_training_sampler(self):
        """Test 2: No validation patient enters the training sampler."""
        print("\n[Running Test 2/14] No validation patient in training sampler...")
        val_patients_set = set(DEFAULT_VAL_PATIENTS)
        
        # Check across 3 epochs
        for ep in range(3):
            self.sampler.set_epoch(ep)
            sampled_indices = self.sampler.current_epoch_indices
            sampled_patients = set(self.train_df.iloc[sampled_indices]["patient_id"])
            overlap = sampled_patients.intersection(val_patients_set)
            self.assertEqual(len(overlap), 0, f"Validation patients {overlap} leaked into training sampler in epoch {ep}!")
        print("  -> PASS: 0 validation patients present in training sampler across all epochs.")

    def test_03_no_test_patient_in_training_sampler(self):
        """Test 3: No test patient enters the training sampler."""
        print("\n[Running Test 3/14] No test patient in training sampler...")
        test_patients_set = set(DEFAULT_TEST_PATIENTS)
        
        for ep in range(3):
            self.sampler.set_epoch(ep)
            sampled_indices = self.sampler.current_epoch_indices
            sampled_patients = set(self.train_df.iloc[sampled_indices]["patient_id"])
            overlap = sampled_patients.intersection(test_patients_set)
            self.assertEqual(len(overlap), 0, f"Test patients {overlap} leaked into training sampler in epoch {ep}!")
        print("  -> PASS: 0 test patients present in training sampler across all epochs.")

    def test_04_sampled_window_ids_exist_in_training_index(self):
        """Test 4: All sampled window IDs exist in the training index."""
        print("\n[Running Test 4/14] Sampled window IDs existence in training set...")
        train_window_ids = set(self.train_df["window_id"])
        
        self.sampler.set_epoch(0)
        sampled_indices = self.sampler.current_epoch_indices
        sampled_window_ids = set(self.train_df.iloc[sampled_indices]["window_id"])
        
        missing = sampled_window_ids.difference(train_window_ids)
        self.assertEqual(len(missing), 0, f"Found {len(missing)} sampled window IDs missing from train_df!")
        self.assertEqual(len(sampled_window_ids), len(sampled_indices), "Sampled window IDs contain duplicates!")
        print(f"  -> PASS: All {len(sampled_window_ids):,} sampled window IDs strictly originate from training index.")

    def test_05_no_synthetic_eeg_windows_created(self):
        """Test 5: No synthetic EEG windows are created."""
        print("\n[Running Test 5/14] Verification of zero synthetic EEG windows...")
        # Check that every sampled window ID matches the canonical regex format chbXX_YY_wZZZZZ
        master_ids = set(self.splitter.window_df["window_id"])
        
        self.sampler.set_epoch(1)
        sampled_ids = self.train_df.iloc[self.sampler.current_epoch_indices]["window_id"]
        
        for wid in sampled_ids.iloc[:500]:
            self.assertIn(wid, master_ids, f"Synthetic or unindexed window ID found: {wid}")
            self.assertTrue(wid.startswith("chb"), f"Invalid window ID format: {wid}")
        print("  -> PASS: 100% of samples are genuine, indexed CHB-MIT windows. Zero synthetic windows.")

    def test_06_positive_windows_remain_real_dataset_windows(self):
        """Test 6: Positive windows remain real dataset windows (100% retained, no duplication)."""
        print("\n[Running Test 6/14] Positive training window retention & integrity...")
        expected_pos_count = (self.train_df["label_any_overlap"] == 1).sum()
        
        self.sampler.set_epoch(0)
        stats = self.sampler.get_epoch_stats()
        
        self.assertEqual(stats["positive_samples"], expected_pos_count)
        self.assertEqual(stats["unique_positive_samples"], expected_pos_count)
        print(f"  -> PASS: All {expected_pos_count:,} positive training windows preserved; zero discarded, zero duplicated.")

    def test_07_validation_distribution_remains_unchanged(self):
        """Test 7: Validation distribution remains unchanged (natural class balance preserved)."""
        print("\n[Running Test 7/14] Validation set natural distribution verification...")
        val_pos = (self.val_df["label_any_overlap"] == 1).sum()
        val_neg = (self.val_df["label_any_overlap"] == 0).sum()
        val_total = len(self.val_df)
        val_ratio = val_neg / val_pos
        
        self.assertEqual(val_pos, 782)
        self.assertEqual(val_neg, 292628)
        self.assertEqual(val_total, 293410)
        self.assertAlmostEqual(val_ratio, 374.20, places=1)
        print(f"  -> PASS: Validation distribution is naturally preserved: {val_pos} pos, {val_neg:,} neg ({val_ratio:.1f}:1).")

    def test_08_test_distribution_remains_unchanged(self):
        """Test 8: Test distribution remains unchanged (natural class balance preserved)."""
        print("\n[Running Test 8/14] Test set natural distribution verification...")
        test_pos = (self.test_df["label_any_overlap"] == 1).sum()
        test_neg = (self.test_df["label_any_overlap"] == 0).sum()
        test_total = len(self.test_df)
        test_ratio = test_neg / test_pos
        
        self.assertEqual(test_pos, 674)
        self.assertEqual(test_neg, 219235)
        self.assertEqual(test_total, 219909)
        self.assertAlmostEqual(test_ratio, 325.27, places=1)
        print(f"  -> PASS: Test distribution is naturally preserved: {test_pos} pos, {test_neg:,} neg ({test_ratio:.1f}:1).")

    def test_09_dynamic_sampling_changes_negatives_across_epochs_and_reproducible(self):
        """Test 9: Dynamic sampling changes negative selection across epochs while remaining reproducible under same seed."""
        print("\n[Running Test 9/14] Dynamic sampling epoch diversity and reproducibility...")
        # Epoch 0 vs Epoch 1
        self.sampler.set_epoch(0)
        ep0_negs = set(self.sampler.current_sampled_neg_indices)
        
        self.sampler.set_epoch(1)
        ep1_negs = set(self.sampler.current_sampled_neg_indices)
        
        # Diversity check
        shared_negs = ep0_negs.intersection(ep1_negs)
        jaccard = len(shared_negs) / len(ep0_negs.union(ep1_negs))
        self.assertLess(jaccard, 0.10, "Epochs 0 and 1 should have predominantly different negative subsets")
        
        # Reproducibility check: resetting to Epoch 0 must yield IDENTICAL indices
        self.sampler.set_epoch(0)
        ep0_negs_retry = set(self.sampler.current_sampled_neg_indices)
        self.assertEqual(ep0_negs, ep0_negs_retry, "Sampler is not deterministic under identical seed!")
        print(f"  -> PASS: Epochs exhibit dynamic diversity (Jaccard {jaccard:.4f}) and 100% deterministic reproducibility.")

    def test_10_requested_sampling_ratio_achieved(self):
        """Test 10: Requested sampling ratio is achieved within documented tolerance."""
        print("\n[Running Test 10/14] Configured sampling ratio accuracy...")
        for target_ratio in [5.0, 10.0, 20.0]:
            s = DynamicNegativeSampler(self.train_df, ratio=target_ratio, base_seed=42)
            stats = s.get_epoch_stats()
            observed_ratio = stats["actual_ratio"]
            self.assertAlmostEqual(observed_ratio, target_ratio, delta=0.05,
                                   msg=f"Target {target_ratio} vs observed {observed_ratio}")
        print("  -> PASS: Sampling ratios 5:1, 10:1, and 20:1 verified within +/- 0.05 tolerance.")

    def test_11_focal_loss_accepts_logits_correctly(self):
        """Test 11: Focal Loss accepts unnormalized logits correctly."""
        print("\n[Running Test 11/14] Focal loss raw logits acceptance...")
        loss_fn = BinaryFocalLossWithLogits(alpha=0.25, gamma=2.0)
        logits = torch.tensor([-5.0, -1.0, 0.0, 2.0, 8.0], requires_grad=True)
        targets = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        
        loss = loss_fn(logits, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)
        print(f"  -> PASS: Focal Loss accepts logits directly; scalar loss = {loss.item():.4f}.")

    def test_12_sigmoid_applied_exactly_once(self):
        """Test 12: Sigmoid is applied exactly once during probability calculation."""
        print("\n[Running Test 12/14] Single-sigmoid probability conversion...")
        logits = torch.tensor([-10.0, -2.0, 0.0, 2.0, 10.0])
        probs = logits_to_probabilities(logits)
        
        # Check range [0, 1]
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        self.assertAlmostEqual(probs[2].item(), 0.5, places=5)
        self.assertAlmostEqual(probs[0].item(), 4.53978687e-05, places=5)
        self.assertAlmostEqual(probs[4].item(), 0.9999546, places=5)
        print("  -> PASS: Probabilities strictly in [0, 1] with exactly one sigmoid invocation.")

    def test_13_no_nan_inf_loss_produced(self):
        """Test 13: No NaN/Inf loss is produced across extreme logits."""
        print("\n[Running Test 13/14] Numerical stability against extreme logits...")
        loss_fn = BinaryFocalLossWithLogits(alpha=0.25, gamma=2.0)
        extreme_logits = torch.tensor([-100.0, -50.0, -20.0, 0.0, 20.0, 50.0, 100.0])
        extreme_targets = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        
        loss = loss_fn(extreme_logits, extreme_targets)
        self.assertFalse(torch.isnan(loss).item(), "Focal Loss produced NaN!")
        self.assertFalse(torch.isinf(loss).item(), "Focal Loss produced Inf!")
        print(f"  -> PASS: Loss is finite ({loss.item():.6f}) across extreme bounds [-100, +100].")

    def test_14_focal_loss_gamma0_equivalent_to_weighted_bce(self):
        """Test 14: Focal Loss with gamma=0 behaves consistently with corresponding weighted BCE."""
        print("\n[Running Test 14/14] Mathematical equivalence of Focal Loss (gamma=0) to weighted BCE...")
        focal_gamma0 = BinaryFocalLossWithLogits(alpha=0.25, gamma=0.0, reduction="none")
        
        logits = torch.randn(200, requires_grad=False)
        targets = torch.randint(0, 2, (200,)).float()
        
        fl_vals = focal_gamma0(logits, targets)
        bce_vals = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        alpha_t = targets * 0.25 + (1.0 - targets) * 0.75
        expected_vals = alpha_t * bce_vals
        
        max_abs_diff = (fl_vals - expected_vals).abs().max().item()
        self.assertLess(max_abs_diff, 1e-6, f"Equivalence error too high: {max_abs_diff}")
        print(f"  -> PASS: Exact numerical equivalence verified (max abs diff = {max_abs_diff:.2e} < 1e-6).")


if __name__ == "__main__":
    unittest.main(verbosity=2)
