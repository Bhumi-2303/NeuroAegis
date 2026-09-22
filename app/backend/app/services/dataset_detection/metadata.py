from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    id: str
    enabled: bool
    path: str
    default_model: str
    display_name: str
    expected_channels: int
    sampling_rate: float
    window_length: int
    feature_count: int
    supported_extensions: list[str]
    channel_count_min: int | None = None
    channel_count_max: int | None = None
    allowed_channel_counts: list[int] = Field(default_factory=list)
    sampling_rate_tolerance: float = 1.0
    filename_patterns: list[str] = Field(default_factory=list)
