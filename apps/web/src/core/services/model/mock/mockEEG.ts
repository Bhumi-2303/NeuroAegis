import type { GraphDataPoint } from '@neuroaegis/model-contracts';

export function generateMockEEGWindow(
  channelIds: string[],
  windowSizeMs: number = 1000,
  samplingRateHz: number = 256
): GraphDataPoint[] {
  const points: GraphDataPoint[] = [];
  const numSamples = Math.floor((windowSizeMs / 1000) * samplingRateHz);
  const now = Date.now();
  const timeStep = 1000 / samplingRateHz;

  for (let i = 0; i < numSamples; i++) {
    const timestampMs = now - windowSizeMs + i * timeStep;
    const timestamp = new Date(timestampMs).toISOString();

    channelIds.forEach((channel, index) => {
      const t = timestampMs / 1000;
      const baseFreq = 10 + (index % 5);
      const slowFreq = 1 + (index % 3);
      
      const value = 
        Math.sin(t * Math.PI * 2 * baseFreq) * 20 + 
        Math.sin(t * Math.PI * 2 * slowFreq) * 30 + 
        (Math.random() - 0.5) * 15;
        
      points.push({
        timestamp,
        value,
        channel
      });
    });
  }

  return points;
}
