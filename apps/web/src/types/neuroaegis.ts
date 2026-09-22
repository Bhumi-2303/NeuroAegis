export type ConfidenceBand = 'low' | 'medium' | 'high';
export type SeizureLabel = 'seizure' | 'non_seizure';
export type AlertSeverity = 'critical' | 'warning' | 'info';
export type MLModelArchitecture = 'random_forest' | 'xgboost' | 'lightgbm';

export interface EegChannelData {
  readonly id: string;
  readonly name: string;
  readonly voltageMicrovolts: number[];
  readonly baselineOffset: number;
}

export interface ShapFeatureContribution {
  readonly featureName: string;
  readonly value: number;
  readonly contribution: number; // positive increases seizure likelihood, negative decreases
  readonly rawValue?: number;
  readonly referenceRange?: readonly [number, number];
}

export interface ShapExplanation {
  readonly baseValue: number;
  readonly features: ShapFeatureContribution[];
}

export interface ModelPrediction {
  readonly modelName: MLModelArchitecture;
  readonly label: SeizureLabel;
  readonly probabilities: {
    readonly seizure: number;
    readonly non_seizure: number;
  };
  readonly confidence: {
    readonly value: number;
    readonly band: ConfidenceBand;
  };
  readonly explanation: ShapExplanation;
  readonly generatedAt: string;
}

export interface PatientVitals {
  readonly patientId: string;
  readonly name: string;
  readonly age: number;
  readonly gender: 'Male' | 'Female' | 'Other';
  readonly heartRateBpm: number;
  readonly spO2Percentage: number;
  readonly samplingRateHz: number;
  readonly clinicalDiagnosis: string;
  readonly activeMedications: string[];
  readonly electrodeMontage: string;
}

export interface SeizureAlert {
  readonly id: string;
  readonly timestamp: string;
  readonly severity: AlertSeverity;
  readonly probability: number;
  readonly primaryChannel: string;
  readonly durationSeconds: number;
  readonly acknowledged: boolean;
}

export interface ModelThresholdConfig {
  readonly sensitivityThreshold: number; // 0.00 to 1.00
  readonly temporalWindowSeconds: number; // 1 to 10s
  readonly gainMultiplier: number; // 0.5x to 3.0x
  readonly highPassFilterHz: number;
  readonly notchFilterEnabled: boolean;
}

export type ViewLifecycleState =
  | { status: 'loading' }
  | { status: 'error'; message: string; code: string; retryCount: number }
  | { status: 'empty' }
  | { status: 'ready' };
