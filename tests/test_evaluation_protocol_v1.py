"""
Unit Tests for NeuroAegis Evaluation Protocol V1.0
Verifies all 12 mandatory synthetic cases and mathematical invariants.
"""

import unittest
import numpy as np
import pandas as pd

from neuroaegis.eval.protocol_v1_evaluator import (
    evaluate_window_level,
    apply_alarm_protocol_v1,
    evaluate_stream_protocol_v1
)


class TestEvaluationProtocolV1(unittest.TestCase):
    
    def test_case_01_one_seizure_one_correct_alarm(self):
        """CASE 1: One seizure, one correct alarm."""
        # Seizure from 10.0s to 30.0s in recording rec_01
        events_df = pd.DataFrame([{
            "recording_id": "rec_01",
            "seizure_id": "sz_01",
            "start_sec": 10.0,
            "end_sec": 30.0
        }])
        
        # 20 windows (each 5.0s, stride 2.5s -> span 0s to 52.5s)
        # Windows starting at 10.0s, 12.5s, 15.0s are positive
        windows = []
        for i in range(20):
            st = i * 2.5
            en = st + 5.0
            is_sz = 1 if (10.0 <= st <= 20.0) else 0
            prob = 0.90 if is_sz == 1 else 0.05
            windows.append({
                "recording_id": "rec_01",
                "window_start_sec": st,
                "window_end_sec": en,
                "label_50pct_overlap": is_sz,
                "predicted_probability": prob
            })
        pred_df = pd.DataFrame(windows)
        
        res = evaluate_stream_protocol_v1(pred_df, events_df, total_monitoring_hours=1.0)
        self.assertEqual(res["event_metrics"]["detected_events"], 1)
        self.assertEqual(res["event_metrics"]["missed_events"], 0)
        self.assertEqual(res["event_metrics"]["event_sensitivity"], 1.0)
        self.assertEqual(res["alarm_metrics"]["clinical_false_alarm_episodes"], 0)
        self.assertEqual(res["event_metrics"]["mean_onset_delay_sec"], 0.0)
        
    def test_case_02_one_seizure_missed(self):
        """CASE 2: One seizure, missed."""
        events_df = pd.DataFrame([{
            "recording_id": "rec_01",
            "seizure_id": "sz_01",
            "start_sec": 10.0,
            "end_sec": 30.0
        }])
        windows = [{
            "recording_id": "rec_01",
            "window_start_sec": i * 2.5,
            "window_end_sec": i * 2.5 + 5.0,
            "label_50pct_overlap": 1 if (10.0 <= i * 2.5 <= 20.0) else 0,
            "predicted_probability": 0.10  # All below threshold 0.50
        } for i in range(20)]
        pred_df = pd.DataFrame(windows)
        
        res = evaluate_stream_protocol_v1(pred_df, events_df, total_monitoring_hours=1.0)
        self.assertEqual(res["event_metrics"]["detected_events"], 0)
        self.assertEqual(res["event_metrics"]["missed_events"], 1)
        self.assertEqual(res["event_metrics"]["event_sensitivity"], 0.0)
        self.assertEqual(res["alarm_metrics"]["clinical_false_alarm_episodes"], 0)
        self.assertIsNone(res["event_metrics"]["mean_onset_delay_sec"])
        
    def test_case_03_one_false_alarm_outside_seizure(self):
        """CASE 3: One false alarm outside seizure."""
        events_df = pd.DataFrame([{
            "recording_id": "rec_01",
            "seizure_id": "sz_01",
            "start_sec": 100.0,
            "end_sec": 120.0
        }])
        # Transient positive burst at 10.0s to 20.0s (far from seizure)
        windows = [{
            "recording_id": "rec_01",
            "window_start_sec": i * 2.5,
            "window_end_sec": i * 2.5 + 5.0,
            "label_50pct_overlap": 1 if (100.0 <= i * 2.5 <= 115.0) else 0,
            "predicted_probability": 0.85 if (10.0 <= i * 2.5 <= 20.0) else 0.05
        } for i in range(60)]
        pred_df = pd.DataFrame(windows)
        
        res = evaluate_stream_protocol_v1(pred_df, events_df, total_monitoring_hours=1.0)
        self.assertEqual(res["event_metrics"]["detected_events"], 0)
        self.assertEqual(res["alarm_metrics"]["clinical_false_alarm_episodes"], 1)
        self.assertGreater(res["alarm_metrics"]["raw_fp_windows"], 0)
        
    def test_case_04_consecutive_positive_windows_form_one_alarm(self):
        """CASE 4: Multiple consecutive positive windows form ONE alarm."""
        binary_seq = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])
        alarms = apply_alarm_protocol_v1(binary_seq, stride_sec=2.5, win_dur_sec=5.0)
        self.assertEqual(len(alarms), 1)
        # Starts at index 2 (5.0s), ends at index 6 (6*2.5 + 5.0 = 20.0s)
        self.assertEqual(alarms[0][0], 5.0)
        self.assertEqual(alarms[0][1], 20.0)
        
    def test_case_05_nearby_alarms_merge_according_to_protocol(self):
        """CASE 5: Two nearby alarms merge according to protocol (gap <= 15s)."""
        # Alarm 1: indices 2..4 (start 5.0s, end 15.0s)
        # Gap: indices 5..6 (zeros -> gap of 5.0s)
        # Alarm 2: indices 7..9 (start 17.5s, end 27.5s)
        # Gap is 17.5 - 15.0 = 2.5s <= 15.0s -> Must merge!
        binary_seq = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
        alarms = apply_alarm_protocol_v1(binary_seq, stride_sec=2.5, win_dur_sec=5.0, merge_gap_sec=15.0)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0][0], 5.0)
        self.assertEqual(alarms[0][1], 27.5)
        
    def test_case_06_alarm_overlaps_seizure_not_counted_as_fp(self):
        """CASE 6: Alarm overlaps true seizure and should not count as FP."""
        events_df = pd.DataFrame([{
            "recording_id": "rec_01",
            "seizure_id": "sz_01",
            "start_sec": 10.0,
            "end_sec": 30.0
        }])
        windows = [{
            "recording_id": "rec_01",
            "window_start_sec": i * 2.5,
            "window_end_sec": i * 2.5 + 5.0,
            "label_50pct_overlap": 1 if (10.0 <= i * 2.5 <= 25.0) else 0,
            "predicted_probability": 0.90 if (10.0 <= i * 2.5 <= 25.0) else 0.05
        } for i in range(20)]
        pred_df = pd.DataFrame(windows)
        
        res = evaluate_stream_protocol_v1(pred_df, events_df, total_monitoring_hours=2.0)
        self.assertEqual(res["alarm_metrics"]["total_alarm_episodes"], 1)
        self.assertEqual(res["alarm_metrics"]["clinical_false_alarm_episodes"], 0)
        
    def test_case_07_alarm_after_seizure_counted_as_fp(self):
        """CASE 7: Alarm after seizure (beyond allowed delay or non-overlapping)."""
        events_df = pd.DataFrame([{
            "recording_id": "rec_01",
            "seizure_id": "sz_01",
            "start_sec": 10.0,
            "end_sec": 20.0
        }])
        # Alarm at 60.0s to 75.0s (postictal / background)
        windows = [{
            "recording_id": "rec_01",
            "window_start_sec": i * 2.5,
            "window_end_sec": i * 2.5 + 5.0,
            "label_50pct_overlap": 1 if (10.0 <= i * 2.5 <= 15.0) else 0,
            "predicted_probability": 0.85 if (60.0 <= i * 2.5 <= 75.0) else 0.05
        } for i in range(40)]
        pred_df = pd.DataFrame(windows)
        
        res = evaluate_stream_protocol_v1(pred_df, events_df, total_monitoring_hours=1.0)
        self.assertEqual(res["event_metrics"]["detected_events"], 0)
        self.assertEqual(res["alarm_metrics"]["clinical_false_alarm_episodes"], 1)
        
    def test_case_08_recording_boundary_isolation(self):
        """CASE 8: Recording boundary (alarms in separate recordings do not merge)."""
        events_df = pd.DataFrame([])  # No seizures
        # Rec 1 ends with positive windows; Rec 2 starts with positive windows
        w1 = [{
            "recording_id": "rec_01",
            "window_start_sec": i * 2.5,
            "window_end_sec": i * 2.5 + 5.0,
            "label_50pct_overlap": 0,
            "predicted_probability": 0.90 if i >= 8 else 0.05
        } for i in range(12)]
        w2 = [{
            "recording_id": "rec_02",
            "window_start_sec": i * 2.5,
            "window_end_sec": i * 2.5 + 5.0,
            "label_50pct_overlap": 0,
            "predicted_probability": 0.90 if i <= 4 else 0.05
        } for i in range(12)]
        pred_df = pd.DataFrame(w1 + w2)
        
        res = evaluate_stream_protocol_v1(pred_df, events_df, total_monitoring_hours=1.0)
        # Must produce exactly 2 separate clinical false alarms, not 1 merged alarm across recordings!
        self.assertEqual(res["alarm_metrics"]["clinical_false_alarm_episodes"], 2)
        
    def test_case_09_continuous_positive_predictions(self):
        """CASE 9: Continuous positive predictions form 1 alarm per recording."""
        events_df = pd.DataFrame([])
        windows = [{
            "recording_id": "rec_01",
            "window_start_sec": i * 2.5,
            "window_end_sec": i * 2.5 + 5.0,
            "label_50pct_overlap": 0,
            "predicted_probability": 0.99
        } for i in range(50)]
        pred_df = pd.DataFrame(windows)
        
        res = evaluate_stream_protocol_v1(pred_df, events_df, total_monitoring_hours=1.0)
        self.assertEqual(res["alarm_metrics"]["total_alarm_episodes"], 1)
        self.assertEqual(res["alarm_metrics"]["raw_fp_windows"], 50)
        
    def test_case_10_no_positive_predictions(self):
        """CASE 10: No positive predictions."""
        events_df = pd.DataFrame([])
        windows = [{
            "recording_id": "rec_01",
            "window_start_sec": i * 2.5,
            "window_end_sec": i * 2.5 + 5.0,
            "label_50pct_overlap": 0,
            "predicted_probability": 0.01
        } for i in range(50)]
        pred_df = pd.DataFrame(windows)
        
        res = evaluate_stream_protocol_v1(pred_df, events_df, total_monitoring_hours=1.0)
        self.assertEqual(res["alarm_metrics"]["total_alarm_episodes"], 0)
        self.assertEqual(res["alarm_metrics"]["raw_fp_windows"], 0)
        self.assertEqual(res["window_metrics"]["specificity"], 1.0)
        
    def test_case_11_bendr_collapse_sanity_check(self):
        """CASE 11: BENDR-like all-positive prediction stream flags collapsed alert state."""
        events_df = pd.DataFrame([{
            "recording_id": "rec_01",
            "seizure_id": "sz_01",
            "start_sec": 10.0,
            "end_sec": 30.0
        }])
        windows = [{
            "recording_id": "rec_01",
            "window_start_sec": i * 2.5,
            "window_end_sec": i * 2.5 + 5.0,
            "label_50pct_overlap": 1 if (10.0 <= i * 2.5 <= 25.0) else 0,
            "predicted_probability": 0.2127  # Uniform above tau=0.10
        } for i in range(50)]
        pred_df = pd.DataFrame(windows)
        
        res = evaluate_stream_protocol_v1(pred_df, events_df, total_monitoring_hours=1.0, threshold=0.10)
        self.assertTrue(res["alarm_metrics"]["is_collapsed_alert_state"])
        self.assertEqual(res["window_metrics"]["specificity"], 0.0)
        self.assertEqual(res["window_metrics"]["sensitivity"], 1.0)
        self.assertEqual(res["alarm_metrics"]["raw_fp_windows"], len(windows) - int(pred_df["label_50pct_overlap"].sum()))
        
    def test_case_12_two_seizures_with_separate_detections(self):
        """CASE 12: Two seizures with separate detections."""
        events_df = pd.DataFrame([
            {"recording_id": "rec_01", "seizure_id": "sz_01", "start_sec": 10.0, "end_sec": 25.0},
            {"recording_id": "rec_01", "seizure_id": "sz_02", "start_sec": 80.0, "end_sec": 95.0}
        ])
        windows = []
        for i in range(50):
            st = i * 2.5
            is_sz = 1 if ((10.0 <= st <= 20.0) or (80.0 <= st <= 90.0)) else 0
            prob = 0.95 if is_sz == 1 else 0.02
            windows.append({
                "recording_id": "rec_01",
                "window_start_sec": st,
                "window_end_sec": st + 5.0,
                "label_50pct_overlap": is_sz,
                "predicted_probability": prob
            })
        pred_df = pd.DataFrame(windows)
        
        res = evaluate_stream_protocol_v1(pred_df, events_df, total_monitoring_hours=1.0)
        self.assertEqual(res["event_metrics"]["total_events"], 2)
        self.assertEqual(res["event_metrics"]["detected_events"], 2)
        self.assertEqual(res["event_metrics"]["missed_events"], 0)
        self.assertEqual(res["event_metrics"]["event_sensitivity"], 1.0)
        self.assertEqual(res["alarm_metrics"]["total_alarm_episodes"], 2)
        self.assertEqual(res["alarm_metrics"]["clinical_false_alarm_episodes"], 0)


if __name__ == "__main__":
    unittest.main()
