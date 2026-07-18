import { useEffect, useState } from 'react';
import type { GraphDataPoint } from '@neuroaegis/model-contracts';
import { useModelService } from '../../../shared/hooks/useModelService';

export function useEEGStream(channelIds: string[]) {
  const modelService = useModelService();
  const [data, setData] = useState<GraphDataPoint[]>([]);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isActive = true;

    async function streamData() {
      try {
        const stream = modelService.streamEEG(channelIds);
        for await (const chunk of stream) {
          if (!isActive) break;
          // Note: In a production app, we would manage a sliding window for the graph
          setData((prev) => [...prev, ...chunk].slice(-1000));
          setIsLoading(false);
        }
      } catch (err) {
        if (isActive) {
          setError(err instanceof Error ? err : new Error('Stream error'));
          setIsLoading(false);
        }
      }
    }

    streamData();

    return () => {
      isActive = false;
    };
  }, [channelIds, modelService]);

  return { data, error, isLoading };
}
