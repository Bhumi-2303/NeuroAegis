export const API_ENDPOINTS = {
  PREDICTION: '/api/v1/predict',
  MODEL_INFO: '/api/v1/model/info',
  HEALTH: '/api/v1/health',
  // Note: These endpoints remain mock for now as they require streaming hardware
  EEG_STREAM: '/api/v1/model/eeg/stream',
  FREQUENCY_BANDS: '/api/v1/model/frequency-bands',
  ALERTS: '/api/v1/alerts',
  PATIENTS: '/api/v1/patients',
} as const;

export type ApiEndpoint = typeof API_ENDPOINTS[keyof typeof API_ENDPOINTS];
