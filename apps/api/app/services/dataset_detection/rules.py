from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Tuple
from .metadata import DatasetMetadata

class DatasetRule(ABC):
    @abstractmethod
    def evaluate(self, df: pd.DataFrame, provided_fs: float, metadata: DatasetMetadata) -> Tuple[float, list[str]]:
        """
        Evaluate how well the dataframe matches the metadata.
        Returns:
            Tuple[float, list[str]]: A confidence score between 0.0 and 1.0, and a list of matched rules/reasons.
        """
        pass

class ChannelCountRule(DatasetRule):
    def evaluate(self, df: pd.DataFrame, provided_fs: float, metadata: DatasetMetadata) -> Tuple[float, list[str]]:
        actual_channels = len(df.columns)
        if actual_channels == metadata.expected_channels:
            return 1.0, [f"Exact match on channel count ({actual_channels} channels)"]
        # Partial match if close (e.g. 22 vs 23)
        diff = abs(actual_channels - metadata.expected_channels)
        if metadata.expected_channels > 1 and diff <= 2:
            return 0.5, [f"Close match on channel count ({actual_channels} channels vs expected {metadata.expected_channels})"]
        return 0.0, []

class SamplingRateRule(DatasetRule):
    def evaluate(self, df: pd.DataFrame, provided_fs: float, metadata: DatasetMetadata) -> Tuple[float, list[str]]:
        if not provided_fs:
            return 0.0, []
        
        diff = abs(provided_fs - metadata.sampling_rate)
        if diff < 1.0:
            return 1.0, [f"Exact match on sampling rate ({provided_fs} Hz)"]
        # CHB-MIT is 256, Bonn is 173.61. A user might pass 256 for Bonn by mistake.
        # But if they pass exactly 173.61, it's a strong indicator.
        if diff < 10.0:
            return 0.5, [f"Close match on sampling rate ({provided_fs} Hz vs expected {metadata.sampling_rate})"]
        return 0.0, []

class SignalLengthRule(DatasetRule):
    def evaluate(self, df: pd.DataFrame, provided_fs: float, metadata: DatasetMetadata) -> Tuple[float, list[str]]:
        actual_samples = len(df)
        if actual_samples == metadata.window_length:
            return 1.0, [f"Exact match on signal length ({actual_samples} samples)"]
            
        # Often EEG data can be slightly longer or a multiple
        if actual_samples > 0 and actual_samples % metadata.window_length == 0:
            return 0.8, [f"Signal length is a multiple of expected window ({actual_samples} samples)"]
            
        # Or it might be close
        if abs(actual_samples - metadata.window_length) < 500:
            return 0.5, [f"Close match on signal length ({actual_samples} samples)"]
            
        return 0.0, []

class DefaultDatasetScorer:
    """Combines rules to produce a final confidence score."""
    def __init__(self):
        self.rules = [
            (ChannelCountRule(), 0.60),  # Channel count is the strongest indicator
            (SamplingRateRule(), 0.20),
            (SignalLengthRule(), 0.20)
        ]
        
    def score(self, df: pd.DataFrame, provided_fs: float, metadata: DatasetMetadata) -> Tuple[float, list[str]]:
        total_score = 0.0
        all_reasons = []
        for rule, weight in self.rules:
            score, reasons = rule.evaluate(df, provided_fs, metadata)
            total_score += score * weight
            all_reasons.extend(reasons)
            
        return total_score, all_reasons
