from __future__ import annotations
from threading import Lock

from apps.api.app.domain.protocols import (
    DatasetDetectorProtocol,
    ExplainerProtocol,
    FeatureExtractorProtocol,
    ModelProtocol,
)

from .base_registry import BaseRegistry


class RegistryHub:
    """
    Singleton Hub that composes and initializes all domain registries.
    Serves as the central lookup service for the Pipeline Orchestrator.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls) -> "RegistryHub":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize_registries()
        return cls._instance

    def _initialize_registries(self) -> None:
        """Initialize empty registries for all plugin types."""
        self.models = BaseRegistry[ModelProtocol]("ModelRegistry")
        self.features = BaseRegistry[FeatureExtractorProtocol]("FeatureRegistry")
        self.explainers = BaseRegistry[ExplainerProtocol]("ExplainerRegistry")
        self.datasets = BaseRegistry[DatasetDetectorProtocol]("DatasetRegistry")

    @classmethod
    def reset(cls) -> None:
        """Testing utility to reset the singleton state."""
        with cls._lock:
            cls._instance = None
