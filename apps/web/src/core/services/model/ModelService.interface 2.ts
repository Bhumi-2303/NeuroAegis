import type {
  ModelInput,
  ModelOutput,
  GraphDataPoint,
  FrequencyBandData,
  EvaluationMetrics,
  Alert
} from '@neuroaegis/model-contracts';

export interface IModelService {
  /**
   * Evaluates the current EEG window and returns a classification prediction.
   */
  getPrediction(input: ModelInput): Promise<ModelOutput>;

  /**
   * Streams live EEG data for visualization.
   */
  streamEEG(channelIds: string[]): AsyncIterable<GraphDataPoint[]>;

  /**
   * Retrieves current frequency band power values.
   */
  getFrequencyBands(): Promise<FrequencyBandData>;

  /**
   * Retrieves evaluation metrics for the active model.
   */
  getEvaluationMetrics(modelName: 'random_forest' | 'xgboost' | 'lightgbm'): Promise<EvaluationMetrics>;

  /**
   * Retrieves active system or patient alerts.
   */
  getAlerts(): Promise<Alert[]>;
}
