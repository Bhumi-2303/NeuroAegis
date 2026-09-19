import { EegChannelData, ModelPrediction, PatientVitals, SeizureAlert } from '../types/neuroaegis';

export const EEG_CHANNEL_NAMES: readonly string[] = [
  'FP1-F7',
  'F7-T7',
  'T7-P7',
  'P7-O1',
  'FP1-F3',
  'F3-C3',
  'C3-P3',
  'P3-O1',
  'FP2-F4',
  'F4-C4',
  'C4-P4',
  'P4-O2',
  'FP2-F8',
  'F8-T8',
  'T8-P8',
  'P8-O2',
] as const;

export const INITIAL_PATIENT: PatientVitals = {
  patientId: 'CHB-MIT-01_S03',
  name: 'Eleanor Vance (Monitored)',
  age: 34,
  gender: 'Female',
  heartRateBpm: 78,
  spO2Percentage: 98,
  samplingRateHz: 256,
  clinicalDiagnosis: 'Refractory Focal Epilepsy with Secondary Generalization',
  activeMedications: ['Levetiracetam 1000mg BID', 'Lamotrigine 150mg BID', 'Clobazam 10mg QHS'],
  electrodeMontage: 'International 10-20 Modified Bipolar (Longitudinal Double Banana)',
};

/**
 * Generates continuous synthetic multi-channel EEG signals with authentic 
 * Alpha/Beta rhythms or High-Amplitude Rhythmic Spike-and-Wave Discharges when seizure is triggered.
 */
export function generateEegBuffer(
  sampleCount: number,
  timeOffset: number,
  isSeizureActive: boolean,
  gain: number = 1.0
): EegChannelData[] {
  return EEG_CHANNEL_NAMES.map((channelName, channelIndex) => {
    const voltages: number[] = new Array(sampleCount);
    const phaseShift = channelIndex * 0.45;

    for (let i = 0; i < sampleCount; i++) {
      const t = (timeOffset + i) / 256;

      // Normal background physiological rhythms: Alpha (8-12Hz) + Beta (13-30Hz) + Theta (4-7Hz)
      const alpha = Math.sin(2 * Math.PI * 10 * t + phaseShift) * 18;
      const beta = Math.sin(2 * Math.PI * 22 * t + phaseShift * 1.5) * 8;
      const theta = Math.sin(2 * Math.PI * 5 * t + phaseShift * 0.8) * 14;
      const noise = (Math.random() - 0.5) * 6;

      let sample = (alpha + beta + theta + noise) * gain;

      if (isSeizureActive) {
        // Rhythmic synchronous 3Hz spike-and-wave paroxysm
        const spike = Math.pow(Math.sin(2 * Math.PI * 3 * t + phaseShift), 9) * 95;
        const wave = Math.sin(2 * Math.PI * 3 * t + phaseShift - Math.PI / 4) * 65;
        sample = (spike + wave + noise * 1.5) * gain;
      }

      voltages[i] = Number(sample.toFixed(2));
    }

    return {
      id: `chan-${channelIndex}`,
      name: channelName,
      voltageMicrovolts: voltages,
      baselineOffset: channelIndex * 60 + 35,
    };
  });
}

export function generatePrediction(
  isSeizureActive: boolean,
  sensitivityThreshold: number
): ModelPrediction {
  const baseSeizureProb = isSeizureActive
    ? 0.84 + Math.random() * 0.14
    : 0.03 + Math.random() * 0.12;

  const seizureProbClamped = Math.min(0.99, Math.max(0.01, baseSeizureProb));
  const nonSeizureProb = 1 - seizureProbClamped;
  const isClassifiedSeizure = seizureProbClamped >= sensitivityThreshold;

  const confidenceValue = Number(
    (isClassifiedSeizure ? seizureProbClamped : nonSeizureProb).toFixed(3)
  );

  let band: 'low' | 'medium' | 'high' = 'low';
  if (confidenceValue >= 0.85) band = 'high';
  else if (confidenceValue >= 0.65) band = 'medium';

  return {
    modelName: 'xgboost',
    label: isClassifiedSeizure ? 'seizure' : 'non_seizure',
    probabilities: {
      seizure: Number(seizureProbClamped.toFixed(4)),
      non_seizure: Number(nonSeizureProb.toFixed(4)),
    },
    confidence: {
      value: confidenceValue,
      band,
    },
    explanation: {
      baseValue: 0.08,
      features: [
        {
          featureName: 'Line Length (T7-P7)',
          value: isSeizureActive ? 412.8 : 88.4,
          contribution: isSeizureActive ? 0.38 : -0.12,
        },
        {
          featureName: 'Delta-Theta Power Ratio',
          value: isSeizureActive ? 3.82 : 0.94,
          contribution: isSeizureActive ? 0.29 : -0.08,
        },
        {
          featureName: 'Spectral Entropy',
          value: isSeizureActive ? 0.41 : 0.89,
          contribution: isSeizureActive ? 0.22 : -0.15,
        },
        {
          featureName: 'Wavelet Energy D4',
          value: isSeizureActive ? 784.1 : 120.2,
          contribution: isSeizureActive ? 0.19 : -0.05,
        },
        {
          featureName: 'Cross-Channel Coherence',
          value: isSeizureActive ? 0.92 : 0.23,
          contribution: isSeizureActive ? 0.14 : -0.09,
        },
      ],
    },
    generatedAt: new Date().toISOString(),
  };
}

export function createInitialAlerts(): SeizureAlert[] {
  return [
    {
      id: 'alt-101',
      timestamp: new Date(Date.now() - 1000 * 180).toLocaleTimeString(),
      severity: 'critical',
      probability: 0.94,
      primaryChannel: 'T7-P7',
      durationSeconds: 12,
      acknowledged: true,
    },
    {
      id: 'alt-102',
      timestamp: new Date(Date.now() - 1000 * 60).toLocaleTimeString(),
      severity: 'warning',
      probability: 0.72,
      primaryChannel: 'FP1-F3',
      durationSeconds: 4,
      acknowledged: false,
    },
  ];
}
