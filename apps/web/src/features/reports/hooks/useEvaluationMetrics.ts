import { useQuery } from '@tanstack/react-query';
import { useModelService } from '../../../shared/hooks/useModelService';
import type { EvaluationMetrics } from '@neuroaegis/model-contracts';

export function useEvaluationMetrics(modelName: 'random_forest' | 'xgboost' | 'lightgbm' = 'random_forest') {
  const modelService = useModelService();

  return useQuery<EvaluationMetrics>({
    queryKey: ['evaluationMetrics', modelName],
    queryFn: () => modelService.getEvaluationMetrics(modelName),
    enabled: !!modelName,
  });
}
