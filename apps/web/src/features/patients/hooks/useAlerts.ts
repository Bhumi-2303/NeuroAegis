import { useQuery } from '@tanstack/react-query';
import { useModelService } from '../../../shared/hooks/useModelService';

export function useAlerts() {
  const modelService = useModelService();

  return useQuery({
    queryKey: ['alerts'],
    queryFn: () => modelService.getAlerts(),
    refetchInterval: 5000,
  });
}
