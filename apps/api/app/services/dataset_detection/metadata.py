from pydantic import BaseModel
from typing import List

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
    supported_extensions: List[str]
