"""
NeuroAegis Patient-Level Splitting & Isolation Partitioner
Phase: Class Imbalance Handling Strategy

Enforces strict patient-independent data splitting prior to any class balancing.
Prevents data leakage by isolating patients at the identity boundary.
Validation and test sets preserve their natural clinical class distributions.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np


# Default canonical holdout partition: 16 Train / 4 Validation / 4 Test
DEFAULT_TRAIN_PATIENTS = [
    "chb04", "chb09", "chb11", "chb12", "chb13", "chb14", "chb15", "chb16",
    "chb17", "chb18", "chb19", "chb20", "chb21", "chb22", "chb23", "chb24"
]
DEFAULT_VAL_PATIENTS = ["chb06", "chb07", "chb08", "chb10"]
DEFAULT_TEST_PATIENTS = ["chb01", "chb02", "chb03", "chb05"]


class PatientDataSplitter:
    """
    Partitions CHB-MIT window index strictly by patient identity.
    Guarantees zero patient leakage across train, validation, and test folds.
    """
    
    def __init__(
        self,
        window_index_path: str = "/Volumes/BLACK-BOX/NeuroAegis/research/data/manifests/chbmit_window_index.csv",
        seizure_events_path: str = "/Volumes/BLACK-BOX/NeuroAegis/research/data/manifests/chbmit_seizure_events.csv",
        train_patients: Optional[List[str]] = None,
        val_patients: Optional[List[str]] = None,
        test_patients: Optional[List[str]] = None,
        label_column: str = "label_any_overlap"
    ):
        self.window_index_path = window_index_path
        self.seizure_events_path = seizure_events_path
        self.label_column = label_column
        
        self.train_patients = sorted(list(train_patients or DEFAULT_TRAIN_PATIENTS))
        self.val_patients = sorted(list(val_patients or DEFAULT_VAL_PATIENTS))
        self.test_patients = sorted(list(test_patients or DEFAULT_TEST_PATIENTS))
        
        self._validate_patient_sets()
        self._load_data()
        
    def _validate_patient_sets(self):
        """Verifies that patient subsets are completely pairwise disjoint."""
        set_train = set(self.train_patients)
        set_val = set(self.val_patients)
        set_test = set(self.test_patients)
        
        # Check mutual exclusivity
        tv_overlap = set_train.intersection(set_val)
        if tv_overlap:
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: Patients appear in both Train and Validation: {tv_overlap}")
            
        tt_overlap = set_train.intersection(set_test)
        if tt_overlap:
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: Patients appear in both Train and Test: {tt_overlap}")
            
        vt_overlap = set_val.intersection(set_test)
        if vt_overlap:
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: Patients appear in both Validation and Test: {vt_overlap}")
            
    def _load_data(self):
        """Loads master window index and seizure event annotations."""
        if not os.path.exists(self.window_index_path):
            raise FileNotFoundError(f"Master window index not found: {self.window_index_path}")
            
        # Low memory False to handle mixed types cleanly
        self.window_df = pd.read_csv(self.window_index_path, low_memory=False)
        all_patients = sorted(self.window_df["patient_id"].unique())
        
        # Verify requested patients exist in dataset
        combined_requested = set(self.train_patients + self.val_patients + self.test_patients)
        missing_from_data = combined_requested.difference(set(all_patients))
        if missing_from_data:
            raise ValueError(f"Requested patients do not exist in dataset: {missing_from_data}")
            
        if os.path.exists(self.seizure_events_path):
            self.events_df = pd.read_csv(self.seizure_events_path)
        else:
            self.events_df = None
            
    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extracts train, validation, and test subsets from the master window index.
        Validation and test subsets maintain their completely natural class distribution.
        
        Returns:
            (train_df, val_df, test_df)
        """
        train_mask = self.window_df["patient_id"].isin(self.train_patients)
        val_mask = self.window_df["patient_id"].isin(self.val_patients)
        test_mask = self.window_df["patient_id"].isin(self.test_patients)
        
        train_df = self.window_df[train_mask].copy().reset_index(drop=True)
        val_df = self.window_df[val_mask].copy().reset_index(drop=True)
        test_df = self.window_df[test_mask].copy().reset_index(drop=True)
        
        # Rigorous post-split assertions
        assert set(train_df["patient_id"]).issubset(set(self.train_patients))
        assert set(val_df["patient_id"]).issubset(set(self.val_patients))
        assert set(test_df["patient_id"]).issubset(set(self.test_patients))
        assert len(set(train_df["patient_id"]).intersection(set(val_df["patient_id"]))) == 0
        assert len(set(train_df["patient_id"]).intersection(set(test_df["patient_id"]))) == 0
        assert len(set(val_df["patient_id"]).intersection(set(test_df["patient_id"]))) == 0
        
        return train_df, val_df, test_df

    def get_split_summary(self) -> Dict[str, Any]:
        """
        Generates detailed statistics for each split partition.
        """
        train_df, val_df, test_df = self.get_splits()
        
        def _calc_stats(df: pd.DataFrame, patients: List[str]) -> Dict[str, Any]:
            pos = int((df[self.label_column] == 1).sum())
            neg = int((df[self.label_column] == 0).sum())
            total = len(df)
            ratio = round(neg / pos, 2) if pos > 0 else float("inf")
            pct = round(pos / total * 100, 3) if total > 0 else 0.0
            
            seiz_count = 0
            seiz_dur = 0.0
            if self.events_df is not None:
                sub_events = self.events_df[self.events_df["patient_id"].isin(patients)]
                seiz_count = len(sub_events)
                seiz_dur = float(sub_events["duration_sec"].sum())
                
            return {
                "patients": patients,
                "patient_count": len(patients),
                "total_windows": total,
                "positive_windows": pos,
                "negative_windows": neg,
                "positive_percentage": pct,
                "natural_ratio": ratio,
                "seizure_event_count": seiz_count,
                "total_seizure_duration_sec": seiz_dur
            }
            
        return {
            "train": _calc_stats(train_df, self.train_patients),
            "validation": _calc_stats(val_df, self.val_patients),
            "test": _calc_stats(test_df, self.test_patients),
            "total_windows_all_splits": len(train_df) + len(val_df) + len(test_df),
            "master_dataset_total_windows": len(self.window_df)
        }
        
    def generate_lopo_folds(self) -> List[Dict[str, Any]]:
        """
        Generates 24 Leave-One-Patient-Out (LOPO-CV) cross-validation fold definitions.
        For each fold i, patient i is the test patient, one patient is validation,
        and the remaining 22 patients constitute the training fold.
        """
        all_patients = sorted(self.window_df["patient_id"].unique())
        folds = []
        
        for i, test_p in enumerate(all_patients):
            # Deterministic selection of validation patient: cyclic next patient
            val_p = all_patients[(i + 1) % len(all_patients)]
            train_pts = [p for p in all_patients if p != test_p and p != val_p]
            
            folds.append({
                "fold_id": f"LOPO_Fold_{i+1:02d}_{test_p}",
                "test_patient": test_p,
                "val_patient": val_p,
                "train_patients": train_pts
            })
            
        return folds
