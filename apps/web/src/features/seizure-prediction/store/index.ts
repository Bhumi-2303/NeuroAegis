import { create } from 'zustand';

interface PredictionState {
  threshold: number;
  autoRefresh: boolean;
  setThreshold: (threshold: number) => void;
  setAutoRefresh: (autoRefresh: boolean) => void;
}

export const usePredictionStore = create<PredictionState>((set) => ({
  threshold: 0.8,
  autoRefresh: true,
  setThreshold: (threshold) => set({ threshold }),
  setAutoRefresh: (autoRefresh) => set({ autoRefresh }),
}));
