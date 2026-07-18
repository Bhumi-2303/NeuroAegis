import type { ModelOutput, ConfidenceScore, ShapExplanation } from '@neuroaegis/model-contracts';

const mockShapExplanation: ShapExplanation = {
  baseValue: 0.25,
  features: [
    { featureName: 'gamma_band_power_t3', value: 0.18 },
    { featureName: 'theta_alpha_ratio_f7', value: 0.12 },
    { featureName: 'beta_activity_c3', value: 0.08 },
    { featureName: 'heart_rate_variability', value: -0.05 },
    { featureName: 'delta_band_power_o1', value: -0.03 },
    { featureName: 'muscle_artifact_emg', value: -0.02 }
  ]
};

export function generateMockPrediction(modelName: 'random_forest' | 'xgboost' | 'lightgbm' = 'random_forest'): ModelOutput {
  const isSeizure = Math.random() > 0.8;
  const baseConfidence = isSeizure ? 0.85 + Math.random() * 0.13 : 0.90 + Math.random() * 0.08;
  
  const confidenceScore: ConfidenceScore = {
    value: baseConfidence,
    band: baseConfidence > 0.9 ? 'high' : baseConfidence > 0.75 ? 'medium' : 'low'
  };

  return {
    modelName,
    generatedAt: new Date().toISOString(),
    prediction: {
      label: isSeizure ? 'seizure' : 'non_seizure',
      probabilities: {
        seizure: isSeizure ? baseConfidence : 1 - baseConfidence,
        non_seizure: isSeizure ? 1 - baseConfidence : baseConfidence
      }
    },
    confidence: confidenceScore,
    explanation: mockShapExplanation
  };
}
