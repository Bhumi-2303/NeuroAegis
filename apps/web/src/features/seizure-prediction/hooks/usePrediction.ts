import { useQuery } from '@tanstack/react-query';
import type { ModelInput } from '@neuroaegis/model-contracts';
import { useModelService } from '../../../shared/hooks/useModelService';

export function usePrediction(input: ModelInput | null) {
  const modelService = useModelService();

  return useQuery({
    queryKey: ['prediction', input],
    queryFn: () => {
      if (!input) throw new Error('Input is required');
      return modelService.getPrediction(input);
    },
    enabled: !!input,
  });
}
