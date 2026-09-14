from __future__ import annotations
from typing import Literal

from pydantic import BaseModel


class RocPointSchema(BaseModel):
    fpr: float
    tpr: float

class ConfusionMatrixSchema(BaseModel):
    truePositive: int
    falsePositive: int
    trueNegative: int
    falseNegative: int

class EvaluationMetricsSchema(BaseModel):
    modelName: Literal["random_forest", "xgboost", "lightgbm"]
    accuracy: float
    precision: float
    recall: float
    f1Score: float
    rocAuc: float
    rocCurve: list[RocPointSchema]
    confusionMatrix: ConfusionMatrixSchema

class ReportSchema(BaseModel):
    id: str
    title: str
    summary: str
    createdAt: str
    metrics: list[EvaluationMetricsSchema]
