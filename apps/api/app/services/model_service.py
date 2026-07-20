import os
import json
import joblib
import logging
from typing import Optional, Dict, Any, List

from app.core.config import settings
from app.services.explainer import shap_service
from app.services.preprocessing import preprocess_eeg
from app.services.feature_extraction import extract_all_features
from app.services.feature_selection import select_and_order_features
from app.schemas.prediction import ModelOutputSchema, PredictionResultSchema, ConfidenceScoreSchema, ShapExplanationSchema

logger = logging.getLogger("neuroaegis.model_service")

class ModelService:
    def __init__(self):
        self.model = None
        self.selected_features: List[str] = []
        self.feature_names: List[str] = []
        self.scaler = None
        self.metadata: Dict[str, Any] = {}
        self.is_loaded = False
        
    def load_artifacts(self) -> bool:
        """
        Load all model artifacts exactly once during startup.
        Returns True if successful, False otherwise.
        """
        try:
            model_dir = settings.MODEL_ASSETS_DIR
            
            # Load metadata (Optional but recommended)
            metadata_path = os.path.join(model_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")
            else:
                logger.info(f"metadata.json not found at {metadata_path}, using defaults")
                self.metadata = {"version": "unknown", "dataset": "unknown"}

            # Load selected features (REQUIRED - defines order and subset)
            features_path = os.path.join(model_dir, "selected_features.json")
            if not os.path.exists(features_path):
                logger.error(f"selected_features.json not found at {features_path}. Model load failed.")
                self.is_loaded = False
                return False

            with open(features_path, "r") as f:
                self.selected_features = json.load(f)
            logger.info(f"Loaded {len(self.selected_features)} selected features from {features_path}")

            # Load feature names (Optional - for validation)
            feature_names_path = os.path.join(model_dir, "feature_names.json")
            if os.path.exists(feature_names_path):
                with open(feature_names_path, "r") as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded feature names from {feature_names_path}")
                
            # Load Scaler (Optional - if used during training)
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Successfully loaded Scaler from {scaler_path}")
            else:
                logger.info(f"No scaler found at {scaler_path}, assuming no scaling required")

            # Load LightGBM model (REQUIRED)
            model_path = os.path.join(model_dir, "final_lightgbm_full_dataset.pkl")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}. Starting in degraded mode.")
                self.is_loaded = False
                return False
                
            self.model = joblib.load(model_path)
            logger.info(f"Successfully loaded LightGBM model from {model_path}")
            
            # Initialize SHAP explainer once
            shap_service.initialize(self.model, self.selected_features)
            logger.info("Initialized SHAP TreeExplainer")
            
            self.is_loaded = True
            return True
                
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def get_prediction(self, eeg_data: np.ndarray, channel_names: List[str], fs: float = 256.0) -> ModelOutputSchema:
        """
        Run the exact pipeline: preprocessing -> feature extraction -> strict feature selection -> inference -> SHAP
        """
        import numpy as np
        
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
            
        # 1. Preprocessing (Exact notebook migration required)
        denoised_data = preprocess_eeg(eeg_data)
        
        # 2. Feature Extraction (Exact notebook migration required)
        all_features = extract_all_features(denoised_data, channel_names, fs)
        
        # 3. Strict Feature Validation and Selection
        feature_vector = select_and_order_features(all_features, self.selected_features)
        
        # Apply scaler if present
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector)
            
        # 4. Inference
        prob_seizure = float(self.model.predict(feature_vector)[0])
        prob_non_seizure = 1.0 - prob_seizure
        
        is_seizure = prob_seizure > 0.5
        label = "seizure" if is_seizure else "non_seizure"
        
        # Calculate confidence band
        confidence_val = max(prob_seizure, prob_non_seizure)
        if confidence_val > 0.9:
            band = "high"
        elif confidence_val > 0.75:
            band = "medium"
        else:
            band = "low"
            
        # 5. SHAP Explanation
        explanation_dict = shap_service.explain_prediction(feature_vector, top_n=10)
        
        import datetime
        return ModelOutputSchema(
            modelName="lightgbm",
            prediction=PredictionResultSchema(
                label=label,
                probabilities={"seizure": prob_seizure, "non_seizure": prob_non_seizure}
            ),
            confidence=ConfidenceScoreSchema(
                value=confidence_val,
                band=band
            ),
            explanation=ShapExplanationSchema(**explanation_dict),
            generatedAt=datetime.datetime.utcnow().isoformat() + "Z"
        )

# Singleton instance
ml_model_service = ModelService()
