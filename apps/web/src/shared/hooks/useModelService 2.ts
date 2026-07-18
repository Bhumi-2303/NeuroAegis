import { useMemo } from 'react';
import { modelService } from '../../core/services/model/ModelService';
import type { IModelService } from '../../core/services/model/ModelService.interface';

/**
 * Hook to inject the IModelService instance into feature hooks.
 * This makes it easy to mock/swap the service for testing in the future
 * without changing feature code.
 */
export function useModelService(): IModelService {
  // Currently we use a singleton, but this hook provides an injection point
  // if we ever want to use a React Context for dependency injection.
  return useMemo(() => modelService, []);
}
