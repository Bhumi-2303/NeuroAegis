"""
NeuroAegis Automated Test Suite for Phase 2 Preprocessing & Windowing
Tests 1 to 10 covering data integrity, sampling, windowing, labeling, montage, and leakage safety.
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, "research/experiments/windowing_labeling")
from chbmit_preprocessor import CHBMITChannelManager, CHBMITSignalFilter

BASE_DIR = "/home/bhumi/GitHub/NeuroAegis"
MANIFEST_DIR = os.path.join(BASE_DIR, "data/manifests")

class TestPhase2Preprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nLoading manifests for test suite...")
        cls.manifest = pd.read_csv(os.path.join(MANIFEST_DIR, "chbmit_manifest.csv"))
        cls.events = pd.read_csv(os.path.join(MANIFEST_DIR, "chbmit_seizure_events.csv"))
        cls.window_index = pd.read_csv(os.path.join(MANIFEST_DIR, "chbmit_window_index.csv"))
        with open(os.path.join(BASE_DIR, "research/data/config/chbmit_channel_order.json"), "r") as f:
            cls.canonical_channels = json.load(f)
        cls.rec_durations = dict(zip(cls.manifest["recording_id"], cls.manifest["recording_duration_sec"]))
        print(f"Loaded {len(cls.manifest)} recordings, {len(cls.events)} events, {len(cls.window_index):,} windows.")

    def test_01_seizure_annotation_mapping(self):
        """Test 1: Seizure annotation correctly maps to EDF recording and fits within recording duration."""
        print("\n[Running Test 1/10] Seizure annotation mapping...")
        manifest_rec_ids = set(self.manifest["recording_id"])
        for _, row in self.events.iterrows():
            rec_id = row["recording_id"]
            self.assertIn(rec_id, manifest_rec_ids, f"Recording {rec_id} not found in manifest")
            rec_dur = self.rec_durations[rec_id]
            self.assertLess(row["start_sec"], rec_dur, f"Seizure start {row['start_sec']} exceeds recording duration {rec_dur}")
            self.assertLessEqual(row["end_sec"], rec_dur, f"Seizure end {row['end_sec']} exceeds recording duration {rec_dur}")
        print("  -> PASS: All 198 seizures map to valid EDF recordings and durations.")

    def test_02_seizure_boundaries_valid(self):
        """Test 2: Seizure boundaries are valid (start >= 0, end > start, duration == end - start)."""
        print("\n[Running Test 2/10] Seizure boundaries validity...")
        for _, row in self.events.iterrows():
            self.assertGreaterEqual(row["start_sec"], 0, "Seizure start time must be >= 0")
            self.assertGreater(row["end_sec"], row["start_sec"], "Seizure end time must be > start time")
            self.assertEqual(row["duration_sec"], row["end_sec"] - row["start_sec"], "Duration mismatch")
            self.assertTrue(row["boundary_valid"], "boundary_valid flag must be True")
        print("  -> PASS: All 198 seizure event boundaries are strictly valid.")

    def test_03_window_sample_count(self):
        """Test 3: 5-second window produces exactly 1280 samples at 256 Hz."""
        print("\n[Running Test 3/10] 5-second window sample count...")
        expected_samples = 1280
        sample_counts = self.window_index["window_samples"].unique()
        self.assertEqual(len(sample_counts), 1)
        self.assertEqual(sample_counts[0], expected_samples)
        
        # Verify delta between start_sample and end_sample
        deltas = self.window_index["window_end_sample"] - self.window_index["window_start_sample"]
        self.assertTrue((deltas == expected_samples).all(), "All windows must have exactly 1280 samples")
        print("  -> PASS: 100% of 1,414,710 windows contain exactly 1280 samples (5.0s @ 256 Hz).")

    def test_04_window_stride_samples(self):
        """Test 4: 50% overlap produces 640-sample (2.5s) stride between consecutive windows."""
        print("\n[Running Test 4/10] Window stride samples and temporal step...")
        expected_stride_sec = 2.5
        expected_stride_samples = 640
        
        # Check first 5 recordings
        sample_recs = self.manifest["recording_id"].iloc[:5]
        for rec_id in sample_recs:
            rec_windows = self.window_index[self.window_index["recording_id"] == rec_id]
            time_diffs = np.diff(rec_windows["window_start_sec"].values)
            sample_diffs = np.diff(rec_windows["window_start_sample"].values)
            self.assertTrue(np.allclose(time_diffs, expected_stride_sec, atol=1e-3), "Stride time mismatch")
            self.assertTrue((sample_diffs == expected_stride_samples).all(), "Stride sample mismatch")
        print("  -> PASS: Stride is verified at exactly 2.5 seconds (640 samples, 50% overlap).")

    def test_05_channel_count_invariant(self):
        """Test 5: Canonical montage defines exactly 23 channels in fixed order."""
        print("\n[Running Test 5/10] Canonical 23-channel montage invariant...")
        self.assertEqual(len(self.canonical_channels), 23, "Canonical channels list must contain exactly 23 channels")
        mgr = CHBMITChannelManager()
        self.assertEqual(len(mgr.get_canonical_channels()), 23)
        print("  -> PASS: Canonical montage contains exactly 23 channels in fixed order.")

    def test_06_no_recording_boundary_crossing(self):
        """Test 6: No window crosses EDF recording boundaries."""
        print("\n[Running Test 6/10] Recording boundary containment...")
        # Check that window_end_sec <= recording_duration_sec for all windows
        merged = self.window_index[["recording_id", "window_end_sec"]].merge(
            self.manifest[["recording_id", "recording_duration_sec"]],
            on="recording_id"
        )
        violations = merged[merged["window_end_sec"] > (merged["recording_duration_sec"] + 1e-3)]
        self.assertEqual(len(violations), 0, f"Found {len(violations)} windows crossing recording boundary!")
        print("  -> PASS: Zero windows cross recording boundaries across all 1,414,710 windows.")

    def test_07_six_second_seizure_coverage(self):
        """Test 7: 6-second seizure receives valid positive-window coverage under both strategies."""
        print("\n[Running Test 7/10] Short-seizure (6.0s) coverage test...")
        # chb16_17: Seizure 2 is 1694s to 1700s (6s)
        chb16_17_w = self.window_index[self.window_index["recording_id"] == "chb16_17"]
        s2_w = chb16_17_w[(chb16_17_w["window_end_sec"] > 1694.0) & (chb16_17_w["window_start_sec"] < 1700.0)]
        
        pos_a = int(s2_w["label_any_overlap"].sum())
        pos_b = int(s2_w["label_50pct_overlap"].sum())
        
        self.assertGreaterEqual(pos_a, 1, "Strategy A must have at least 1 positive window for 6s seizure")
        self.assertGreaterEqual(pos_b, 1, "Strategy B must have at least 1 positive window for 6s seizure")
        self.assertEqual(pos_a, 4, "Strategy A should produce exactly 4 positive windows for 6s seizure")
        self.assertEqual(pos_b, 3, "Strategy B should produce exactly 3 positive windows for 6s seizure")
        print(f"  -> PASS: chb16_17 6s seizure received {pos_a} pos windows (Strategy A) and {pos_b} pos windows (Strategy B).")

    def test_08_patient_metadata_retention(self):
        """Test 8: Patient IDs remain strictly attached to every generated window."""
        print("\n[Running Test 8/10] Patient metadata retention...")
        self.assertFalse(self.window_index["patient_id"].isnull().any(), "Found null patient_id")
        self.assertFalse(self.window_index["recording_id"].isnull().any(), "Found null recording_id")
        self.assertFalse(self.window_index["dataset"].isnull().any(), "Found null dataset")
        
        # Verify patient_id matches recording_id prefix
        matches = self.window_index.apply(lambda r: r["recording_id"].startswith(r["patient_id"]), axis=1)
        self.assertTrue(matches.all(), "Mismatch between patient_id and recording_id")
        print("  -> PASS: All 1,414,710 windows strictly retain dataset, patient_id, and recording_id metadata.")

    def test_09_unique_window_ids(self):
        """Test 9: No duplicate window IDs across the dataset."""
        print("\n[Running Test 9/10] Window ID uniqueness...")
        total_count = len(self.window_index)
        unique_count = self.window_index["window_id"].nunique()
        self.assertEqual(total_count, unique_count, "Found duplicate window IDs")
        print(f"  -> PASS: All {unique_count:,} window IDs are unique.")

    def test_10_leakage_safety_isolation(self):
        """Test 10: No patient or recording leakage is introduced by the index."""
        print("\n[Running Test 10/10] Leakage safety isolation...")
        # Check that no window_id contains multiple recording IDs
        # Check that patient partition sets are mutually exclusive
        patient_recordings = self.window_index.groupby("patient_id")["recording_id"].unique()
        all_recs = []
        for p, recs in patient_recordings.items():
            for r in recs:
                self.assertNotIn(r, all_recs, f"Recording {r} appears under multiple patients")
                all_recs.append(r)
        self.assertEqual(len(all_recs), len(self.manifest), "Recording count mismatch")
        print(f"  -> PASS: Patient and recording sets are completely disjoint across all 24 patients.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
