import type { IModelService } from './ModelService.interface';
import type {
  ModelInput,
  ModelOutput,
  GraphDataPoint,
  FrequencyBandData,
  EvaluationMetrics,
  Alert
} from '@neuroaegis/model-contracts';

import { generateMockPrediction } from './mock/mockPredictions';
import { generateMockEEGWindow } from './mock/mockEEG';
import { generateMockFrequencyBands } from './mock/mockFrequencyBands';
import { generateMockEvaluationMetrics } from './mock/mockEvaluationMetrics';
import { generateMockAlerts } from './mock/mockAlerts';

/**
 * Singleton implementation of IModelService.
 * Currently serves deterministic mock data with simulated network latency.
 */
class ModelService implements IModelService {
  private simulateLatency<T>(data: T, delayMs: number = 300): Promise<T> {
    return new Promise((resolve) => setTimeout(() => resolve(data), delayMs));
  }

  async getPrediction(input: ModelInput): Promise<ModelOutput> {
    // TODO: Integrate Trained Model
    // When the real model is integrated, replace this mock generator
    // with an HTTP call to the Python backend inference endpoint.
    const modelName = (input.metadata?.modelName as 'random_forest' | 'xgboost' | 'lightgbm') || 'random_forest';
    return this.simulateLatency(generateMockPrediction(modelName), 400);
  }

  async *streamEEG(channelIds: string[]): AsyncIterable<GraphDataPoint[]> {
    // TODO: Integrate Trained Model / Streaming API
    // Replace with WebSocket or Server-Sent Events connection.
    while (true) {
      yield generateMockEEGWindow(channelIds, 100, 256);
      await new Promise((resolve) => setTimeout(resolve, 100)); // 10Hz updates
    }
  }

  async getFrequencyBands(): Promise<FrequencyBandData> {
    // TODO: Integrate Trained Model
    return this.simulateLatency(generateMockFrequencyBands(), 200);
  }

  async getEvaluationMetrics(modelName: 'random_forest' | 'xgboost' | 'lightgbm'): Promise<EvaluationMetrics> {
    // TODO: Integrate Trained Model
    return this.simulateLatency(generateMockEvaluationMetrics(modelName), 500);
  }

  async getAlerts(): Promise<Alert[]> {
    // TODO: Integrate Trained Model
    return this.simulateLatency(generateMockAlerts(), 300);
  }
}

export const modelService = new ModelService();
