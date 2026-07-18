import { z } from 'zod';

// prediction.types.ts
export const ModelInputSchema = z.object({
  windowData: z.array(z.number()), // Simplified for raw array passing
  modelId: z.string().optional(),
  samplingRate: z.number().optional(),
  channelIds: z.array(z.string()).optional()
});

export const ConfidenceScoreSchema = z.object({
  value: z.number().min(0).max(1),
  level: z.enum(['low', 'medium', 'high'])
});

// explanation.types.ts
export const ShapFeatureContributionSchema = z.object({
  name: z.string(),
  value: z.number(),
  contribution: z.enum(['positive', 'negative'])
});

export const ShapExplanationSchema = z.object({
  baseValue: z.number(),
  features: z.array(ShapFeatureContributionSchema)
});

export const PredictionResultSchema = z.object({
  id: z.string(),
  timestamp: z.number(),
  modelId: z.string(),
  classification: z.enum(['seizure', 'non_seizure', 'artifact', 'baseline']),
  confidence: ConfidenceScoreSchema,
  explanation: ShapExplanationSchema.optional(),
  inferenceTimeMs: z.number().optional()
});

// eeg.types.ts
export const GraphDataPointSchema = z.object({
  timestamp: z.number(),
  values: z.record(z.string(), z.number())
});

export const FrequencyBandDataSchema = z.object({
  band: z.string(),
  range: z.string(),
  power: z.number(),
  trend: z.enum(['up', 'down', 'stable'])
});

// metrics.types.ts
export const ConfusionMatrixSchema = z.object({
  truePositives: z.number(),
  falsePositives: z.number(),
  trueNegatives: z.number(),
  falseNegatives: z.number()
});

export const RocPointSchema = z.object({
  fpr: z.number(),
  tpr: z.number()
});

export const EvaluationMetricsSchema = z.object({
  modelId: z.string(),
  timestamp: z.number(),
  dataset: z.string().optional(),
  accuracy: z.number().min(0).max(1),
  precision: z.number().min(0).max(1),
  recall: z.number().min(0).max(1),
  f1Score: z.number().min(0).max(1),
  rocAuc: z.number().min(0).max(1),
  confusionMatrix: ConfusionMatrixSchema,
  rocCurve: z.array(RocPointSchema).optional()
});

// alerts.types.ts
export const AlertSchema = z.object({
  id: z.string(),
  timestamp: z.number(),
  severity: z.enum(['info', 'warning', 'critical']),
  message: z.string(),
  modelId: z.string().optional()
});
