import { create } from 'zustand';

interface ExplainabilityState {
  selectedFeature: string | null;
  showBaseline: boolean;
  setSelectedFeature: (feature: string | null) => void;
  setShowBaseline: (show: boolean) => void;
}

export const useExplainabilityStore = create<ExplainabilityState>((set) => ({
  selectedFeature: null,
  showBaseline: false,
  setSelectedFeature: (feature) => set({ selectedFeature: feature }),
  setShowBaseline: (show) => set({ showBaseline: show }),
}));
