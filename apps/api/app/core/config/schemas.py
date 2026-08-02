from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class PluginConfig(BaseModel):
    """Configuration for a specific plugin instance."""
    plugin_id: str
    params: Dict[str, Any] = Field(default_factory=dict)

class PipelineConfig(BaseModel):
    """
    Schema for a full ML Pipeline configuration.
    Defines the components and parameters required for an execution run.
    """
    id: str
    task: str
    dataset: str
    model: PluginConfig
    feature_extractors: List[PluginConfig] = Field(min_length=1)
    explainer: Optional[PluginConfig] = None
