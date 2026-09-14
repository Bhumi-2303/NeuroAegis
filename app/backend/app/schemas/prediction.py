from __future__ import annotations
from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictionResultSchema(BaseModel):
    label: Literal["seizure", "non_seizure"]
    probabilities: dict[Literal["seizure", "non_seizure"], float]

class ConfidenceScoreSchema(BaseModel):
    value: float = Field(..., ge=0, le=1.0)
    band: Literal["low", "medium", "high"]

class ShapFeatureContributionSchema(BaseModel):
    featureName: str
    value: float
    rawValue: float | None = None
    referenceRange: list[float] | None = None

class ShapExplanationSchema(BaseModel):
    baseValue: float
    features: list[ShapFeatureContributionSchema]

class ModelInputSchema(BaseModel):
    sessionId: str
    signalWindow: list[float]
    channelIds: list[str]
    samplingRateHz: float
    timestamp: str
    metadata: dict[str, Any] | None = None

class ModelOutputSchema(BaseModel):
    modelName: Literal["random_forest", "xgboost", "lightgbm"]
    prediction: PredictionResultSchema
    confidence: ConfidenceScoreSchema
    explanation: ShapExplanationSchema
    generatedAt: str

class AlertSchema(BaseModel):
    id: str
    severity: Literal["info", "warning", "critical"]
    message: str
    createdAt: str
