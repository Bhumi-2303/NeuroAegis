from __future__ import annotations
import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.model_service import ml_model_service

router = APIRouter()

class WindowData(BaseModel):
    window_idx: int
    target: int
    features: dict[str, float]

class LiveMonitorResponse(BaseModel):
    record: str
    windows: list[WindowData]

class PredictFeaturesRequest(BaseModel):
    features: dict[str, float]

@router.get("/live-monitor-data", response_model=LiveMonitorResponse)
def get_live_monitor_data():
    try:
        parquet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "chbmit_subset.parquet")
        if not os.path.exists(parquet_path):
            raise HTTPException(status_code=404, detail="chbmit_subset.parquet not found")

        df = pd.read_parquet(parquet_path)
        record_name = "chb04_28.edf"
        df_record = df[df['record'] == record_name].sort_values('window_idx')
        
        exclude_cols = {'target', 'patient_id', 'record', 'window_idx'}
        feature_cols = [col for col in df_record.columns if col not in exclude_cols]

        windows = []
        for _, row in df_record.iterrows():
            feat_dict = {col: float(row[col]) for col in feature_cols}
            windows.append(WindowData(
                window_idx=int(row['window_idx']),
                target=int(row['target']),
                features=feat_dict
            ))

        return LiveMonitorResponse(record=record_name, windows=windows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-features")
def predict_features(request: PredictFeaturesRequest):
    if not ml_model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    predictor = ml_model_service.get_predictor('chbmit')
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="CHBMIT model is not loaded")
        
    vector = []
    for feat in predictor.selected_features:
        vector.append(request.features.get(feat, 0.0))
    feature_vector = np.array([vector])

    pred_res = predictor.predict(feature_vector)
    shap_res = predictor.generate_explanation(feature_vector)
    
    return {
        "label": pred_res["label"],
        "probability_seizure": pred_res["probabilities"]["seizure"],
        "shap_explanation": shap_res
    }
