import type { EvaluationMetrics } from '@neuroaegis/model-contracts';

export function generateMockEvaluationMetrics(modelName: 'random_forest' | 'xgboost' | 'lightgbm' = 'random_forest'): EvaluationMetrics {
  const isXG = modelName === 'xgboost';
  const isLGBM = modelName === 'lightgbm';

  return {
    modelName,
    accuracy: isXG ? 0.94 : isLGBM ? 0.95 : 0.92,
    precision: isXG ? 0.93 : isLGBM ? 0.94 : 0.90,
    recall: isXG ? 0.95 : isLGBM ? 0.96 : 0.91,
    f1Score: isXG ? 0.94 : isLGBM ? 0.95 : 0.90,
    rocAuc: isXG ? 0.97 : isLGBM ? 0.98 : 0.95,
    rocCurve: Array.from({ length: 11 }).map((_, i) => ({
      fpr: i / 10,
      tpr: Math.pow(i / 10, 0.5) // mock curve
    })),
    confusionMatrix: {
      truePositive: isXG ? 142 : isLGBM ? 145 : 138,
      falsePositive: isXG ? 10 : isLGBM ? 8 : 15,
      trueNegative: isXG ? 280 : isLGBM ? 285 : 270,
      falseNegative: isXG ? 8 : isLGBM ? 5 : 12
    }
  };
}
