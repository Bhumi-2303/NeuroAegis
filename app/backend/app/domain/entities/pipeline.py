from __future__ import annotations
from typing import Any
from uuid import UUID

from pydantic import BaseModel, model_validator
from typing_extensions import Self


class Pipeline(BaseModel):
    """
    Domain entity representing a specific ML pipeline configuration.
    """
    id: UUID
    task_id: str
    dataset_id: str
    configuration: dict[str, Any]

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if "model" not in self.configuration:
            raise ValueError("Pipeline configuration must specify a model.")
        if "feature_extractors" not in self.configuration or not isinstance(self.configuration["feature_extractors"], list) or len(self.configuration["feature_extractors"]) == 0:
            raise ValueError("Pipeline configuration must specify at least one feature extractor.")
        return self
