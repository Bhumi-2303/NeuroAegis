import { useQuery } from '@tanstack/react-query';
import type { ModelInput } from '@neuroaegis/model-contracts';
import { useModelService } from '../../../shared/hooks/useModelService';

export function useExplanation(input: ModelInput | null) {
  const modelService = useModelService();

  return useQuery({
    queryKey: ['explanation', input],
    queryFn: async () => {
      if (!input) throw new Error('Input is required');
      // For now, explanation uses getPrediction data 
      // as our contract might embed explanation in prediction result later
      return modelService.getPrediction(input);
    },
    enabled: !!input,
  });
}
