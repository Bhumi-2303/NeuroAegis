import { useQuery } from '@tanstack/react-query';
import { useModelService } from '../../../shared/hooks/useModelService';

export function useFrequencyBands() {
  const modelService = useModelService();

  return useQuery({
    queryKey: ['frequencyBands'],
    queryFn: () => modelService.getFrequencyBands(),
    refetchInterval: 1000,
  });
}
