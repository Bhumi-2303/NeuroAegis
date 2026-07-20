from pydantic import BaseModel
from typing import List, Optional

class GraphDataPointSchema(BaseModel):
    timestamp: str
    value: float
    channel: Optional[str] = None

class FrequencyBandDataSchema(BaseModel):
    gamma: List[GraphDataPointSchema]
    beta: List[GraphDataPointSchema]
    alpha: List[GraphDataPointSchema]
    theta: List[GraphDataPointSchema]
    delta: List[GraphDataPointSchema]
