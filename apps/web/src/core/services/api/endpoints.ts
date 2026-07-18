export const API_ENDPOINTS = {
  PREDICTION: '/api/v1/model/prediction',
  EEG_STREAM: '/api/v1/model/eeg/stream',
  FREQUENCY_BANDS: '/api/v1/model/frequency-bands',
  METRICS: '/api/v1/model/metrics',
  ALERTS: '/api/v1/alerts',
  PATIENTS: '/api/v1/patients',
} as const;

export type ApiEndpoint = typeof API_ENDPOINTS[keyof typeof API_ENDPOINTS];
