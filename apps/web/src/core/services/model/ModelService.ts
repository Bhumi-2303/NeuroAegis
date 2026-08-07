import type { IModelService } from './ModelService.interface';
import type {
  ModelInput,
  ModelOutput,
  GraphDataPoint,
  FrequencyBandData,
  EvaluationMetrics,
  Alert
} from '@neuroaegis/model-contracts';


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
    const params = new URLSearchParams();
    params.append('channels', channelIds.join(','));
    params.append('ms_per_window', '100');
    
    // Use the Fetch API to read the SSE stream manually since EventSource doesn't play well with async iterators directly without wrapping.
    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    const response = await fetch(`${API_URL}/api/v1/stream/eeg?${params.toString()}`);
    if (!response.ok || !response.body) {
      throw new Error('Failed to connect to EEG stream');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // Parse SSE lines
        let eolIndex;
        while ((eolIndex = buffer.indexOf('\n\n')) >= 0) {
          const chunk = buffer.slice(0, eolIndex);
          buffer = buffer.slice(eolIndex + 2);
          
          if (chunk.startsWith('data: ')) {
            const dataStr = chunk.slice(6);
            if (dataStr.trim()) {
              try {
                const points = JSON.parse(dataStr) as GraphDataPoint[];
                yield points;
              } catch (e) {
                console.error("Failed to parse SSE chunk", e);
              }
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
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
