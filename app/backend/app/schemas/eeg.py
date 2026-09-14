from __future__ import annotations

from pydantic import BaseModel


class GraphDataPointSchema(BaseModel):
    timestamp: str
    value: float
    channel: str | None = None

class FrequencyBandDataSchema(BaseModel):
    gamma: list[GraphDataPointSchema]
    beta: list[GraphDataPointSchema]
    alpha: list[GraphDataPointSchema]
    theta: list[GraphDataPointSchema]
    delta: list[GraphDataPointSchema]
