import { useEffect, useState, useMemo } from 'react';
import type { GraphDataPoint, Alert } from '@neuroaegis/model-contracts';
import { useModelService } from '../../../shared/hooks';
import { useQuery } from '@tanstack/react-query';

/**
 * Dashboard-specific hook that provides a summary of EEG data
 * for the hero neural signal sparkline. Avoids cross-feature import.
 */
export function useDashboardEEG(channelIds: string[]) {
  const modelService = useModelService();
  const [data, setData] = useState<GraphDataPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isActive = true;

    async function streamData() {
      try {
        const stream = modelService.streamEEG(channelIds);
        for await (const chunk of stream) {
          if (!isActive) break;
          setData((prev) => [...prev, ...chunk].slice(-200));
          setIsLoading(false);
        }
      } catch {
        if (isActive) setIsLoading(false);
      }
    }

    streamData();
    return () => { isActive = false; };
  }, [channelIds, modelService]);

  const sparklineValues = useMemo(() => {
    if (!data || data.length === 0) return [];
    return data.slice(-60).map((p) => p.value);
  }, [data]);

  return { sparklineValues, isLoading };
}

/**
 * Dashboard-specific hook for alert count.
 * Avoids cross-feature import from patients.
 */
export function useDashboardAlerts() {
  const modelService = useModelService();

  const query = useQuery<Alert[]>({
    queryKey: ['dashboard-alerts'],
    queryFn: () => modelService.getAlerts(),
    refetchInterval: 10000,
  });

  const criticalCount = useMemo(() => {
    return query.data?.filter((a) => a.severity === 'critical').length ?? 0;
  }, [query.data]);

  return { criticalCount, isLoading: query.isLoading };
}
