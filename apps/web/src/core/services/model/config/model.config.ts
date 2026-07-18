export const MODEL_CONFIG = {
  // Toggle this when a real backend is available
  USE_MOCK_DATA: true,

  // Configurable latency for mock responses to simulate network/inference delay
  MOCK_LATENCY_MIN_MS: 300,
  MOCK_LATENCY_MAX_MS: 900,
  
  // Optional flag to simulate random network/inference errors
  SIMULATE_ERRORS: false,
  ERROR_PROBABILITY: 0.05,

  // Model metadata
  AVAILABLE_MODELS: [
    { id: 'rf-01', name: 'Random Forest v1.2' },
    { id: 'xgb-02', name: 'XGBoost Clinical' },
    { id: 'lgb-01', name: 'LightGBM Fast' }
  ],

  // Clinical thresholds
  THRESHOLDS: {
    HIGH_CONFIDENCE: 0.90,
    MEDIUM_CONFIDENCE: 0.75
  },

  // EEG visual config
  EEG: {
    DEFAULT_CHANNELS: ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'],
    SAMPLE_RATE: 256,
    WINDOW_SIZE_MS: 2000
  }
} as const;
