from __future__ import annotations
from typing import Any

from pydantic import BaseModel, Field


class PluginConfig(BaseModel):
    """Configuration for a specific plugin instance."""
    plugin_id: str
    params: dict[str, Any] = Field(default_factory=dict)

class PipelineConfig(BaseModel):
    """
    Schema for a full ML Pipeline configuration.
    Defines the components and parameters required for an execution run.
    """
    id: str
    task: str
    dataset: str
    model: PluginConfig
    feature_extractors: list[PluginConfig] = Field(min_length=1)
    explainer: PluginConfig | None = None
