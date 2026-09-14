from __future__ import annotations
import logging

import pandas as pd

from app.core.config import settings

from .metadata import DatasetMetadata
from .rules import DefaultDatasetScorer
from .validator import DatasetValidator

logger = logging.getLogger("neuroaegis.dataset_detector")

class DatasetDetector:
    def __init__(self):
        self.scorer = DefaultDatasetScorer()
        self.datasets: dict[str, DatasetMetadata] = {}
        self._load_metadata()

    def _load_metadata(self):
        config = settings.MODELS_CONFIG.get("models", {})
        for dataset_id, cfg in config.items():
            if not cfg.get("enabled", False):
                continue
                
            try:
                # Add sensible defaults if missing
                meta = DatasetMetadata(
                    id=dataset_id,
                    enabled=cfg.get("enabled", True),
                    path=cfg.get("path", ""),
                    default_model=cfg.get("default_model", "lightgbm"),
                    display_name=cfg.get("display_name", dataset_id),
                    expected_channels=cfg.get("expected_channels", 1),
                    sampling_rate=cfg.get("sampling_rate", 256.0),
                    window_length=cfg.get("window_length", 4097),
                    feature_count=cfg.get("feature_count", 0),
                    supported_extensions=cfg.get("supported_extensions", [".csv"])
                )
                self.datasets[dataset_id] = meta
            except Exception as e:
                logger.error(f"Failed to load metadata for dataset {dataset_id}: {e}")

    def detect(self, df: pd.DataFrame, provided_fs: float = 0.0) -> tuple[str, float, list[str]]:
        """
        Validates the dataset and detects which dataset pipeline it belongs to.
        Returns:
            Tuple[str, float, List[str]]: detected_dataset_id, confidence (0 to 1), reasons
        """
        is_valid, msg = DatasetValidator.validate(df)
        if not is_valid:
            raise ValueError(f"Dataset validation failed: {msg}")

        best_dataset = None
        best_score = -1.0
        best_reasons = []

        for dataset_id, metadata in self.datasets.items():
            score, reasons = self.scorer.score(df, provided_fs, metadata)
            if score > best_score:
                best_score = score
                best_dataset = dataset_id
                best_reasons = reasons

        if not best_dataset:
            raise ValueError("Unable to determine EEG dataset.")

        return best_dataset, best_score, best_reasons

# Singleton instance
dataset_detector = DatasetDetector()
