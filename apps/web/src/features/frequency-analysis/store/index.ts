import { create } from 'zustand';

interface FrequencyState {
  selectedBand: string | null;
  setSelectedBand: (band: string | null) => void;
}

export const useFrequencyStore = create<FrequencyState>((set) => ({
  selectedBand: null,
  setSelectedBand: (band) => set({ selectedBand: band }),
}));
