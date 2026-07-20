from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

class PredictionResultSchema(BaseModel):
    label: Literal["seizure", "non_seizure"]
    probabilities: Dict[Literal["seizure", "non_seizure"], float]

class ConfidenceScoreSchema(BaseModel):
    value: float = Field(..., ge=0, le=1.0)
    band: Literal["low", "medium", "high"]

class ShapFeatureContributionSchema(BaseModel):
    featureName: str
    value: float

class ShapExplanationSchema(BaseModel):
    baseValue: float
    features: List[ShapFeatureContributionSchema]

class ModelInputSchema(BaseModel):
    sessionId: str
    signalWindow: List[float]
    channelIds: List[str]
    samplingRateHz: float
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

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
