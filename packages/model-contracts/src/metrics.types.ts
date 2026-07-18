export interface EvaluationMetrics {
  modelName: "random_forest" | "xgboost" | "lightgbm";
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  rocCurve: { fpr: number; tpr: number }[];
  confusionMatrix: {
    truePositive: number;
    falsePositive: number;
    trueNegative: number;
    falseNegative: number;
  };
}

export interface Report {
  id: string;
  title: string;
  summary: string;
  createdAt: string;
  metrics: EvaluationMetrics[];   // one entry per compared model
}
