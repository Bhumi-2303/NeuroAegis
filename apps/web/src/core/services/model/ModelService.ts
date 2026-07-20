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
import { API_ENDPOINTS } from '../api/endpoints';

/**
 * Singleton implementation of IModelService.
 * Currently serves deterministic mock data with simulated network latency.
 */
class ModelService implements IModelService {
  private simulateLatency<T>(data: T, delayMs: number = 300): Promise<T> {
    return new Promise((resolve) => setTimeout(() => resolve(data), delayMs));
  }

  async getPrediction(input: ModelInput): Promise<ModelOutput> {
    const csvHeader = input.channelIds.join(',');
    
    // For simplicity, we chunk the flat signalWindow array by channel count
    // Assuming signalWindow is flattened [ch1_t1, ch2_t1, ch1_t2, ch2_t2...]
    const numChannels = input.channelIds.length;
    const rows: string[] = [];
    
    for (let i = 0; i < input.signalWindow.length; i += numChannels) {
      const row = input.signalWindow.slice(i, i + numChannels).join(',');
      rows.push(row);
    }
    
    const csvContent = `${csvHeader}\n${rows.join('\n')}`;
    const blob = new Blob([csvContent], { type: 'text/csv' });
    
    const formData = new FormData();
    formData.append('file', blob, 'eeg.csv');
    formData.append('sampling_rate', input.samplingRateHz.toString());
    formData.append('channels', input.channelIds.join(','));

    const response = await fetch(API_ENDPOINTS.PREDICTION, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Prediction API failed: ${response.statusText}`);
    }

    return response.json();
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
    const response = await fetch(API_ENDPOINTS.MODEL_INFO);
    if (!response.ok) {
      throw new Error(`Model Info API failed: ${response.statusText}`);
    }
    
    // The backend provides model info, but the UI expects full EvaluationMetrics.
    // For now, we fetch the info to ensure backend is alive, and merge with mock metrics
    // to preserve the rich UI charts (ROC, Confusion Matrix) which might not be stored in the model metadata yet.
    await response.json();
    
    return this.simulateLatency(generateMockEvaluationMetrics(modelName), 200);
  }

  async getAlerts(): Promise<Alert[]> {
    // TODO: Integrate Trained Model
    return this.simulateLatency(generateMockAlerts(), 300);
  }
}

export const modelService = new ModelService();
