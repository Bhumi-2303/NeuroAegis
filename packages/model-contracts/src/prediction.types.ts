export interface ModelInput {
  sessionId: string;
  signalWindow: number[];         // preprocessed/segmented EEG samples for this window
  channelIds: string[];
  samplingRateHz: number;
  timestamp: string;              // ISO
  metadata?: Record<string, unknown>;
}

export interface PredictionResult {
  label: "seizure" | "non_seizure";
  probabilities: {
    seizure: number;
    non_seizure: number;
  };
}

export interface ConfidenceScore {
  value: number;                  // 0–1
  band: "low" | "medium" | "high";
}

import type { ShapExplanation } from "./explanation.types";

export interface ModelOutput {
  modelName: "random_forest" | "xgboost" | "lightgbm";
  prediction: PredictionResult;
  confidence: ConfidenceScore;
  explanation: ShapExplanation;
  generatedAt: string;
}
