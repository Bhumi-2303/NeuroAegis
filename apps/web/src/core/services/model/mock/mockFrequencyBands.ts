import type { FrequencyBandData, GraphDataPoint } from '@neuroaegis/model-contracts';

export function generateMockFrequencyBands(): FrequencyBandData {
  const now = Date.now();
  
  const createBandData = (baseVal: number, variance: number): GraphDataPoint[] => {
    return Array.from({ length: 60 }).map((_, i) => ({
      timestamp: new Date(now - (59 - i) * 1000).toISOString(),
      value: Math.max(0, baseVal + (Math.random() - 0.5) * variance)
    }));
  };

  return {
    gamma: createBandData(15, 5),
    beta: createBandData(25, 10),
    alpha: createBandData(45, 15),
    theta: createBandData(20, 8),
    delta: createBandData(10, 4)
  };
}
