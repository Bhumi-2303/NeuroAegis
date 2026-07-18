import { create } from 'zustand';

interface EegState {
  selectedChannels: string[];
  timeWindow: number;
  isRunning: boolean;
  setSelectedChannels: (channels: string[]) => void;
  setTimeWindow: (window: number) => void;
  setIsRunning: (isRunning: boolean) => void;
}

export const useEegStore = create<EegState>((set) => ({
  selectedChannels: ['Fp1', 'Fp2', 'C3', 'C4'],
  timeWindow: 10,
  isRunning: true,
  setSelectedChannels: (channels) => set({ selectedChannels: channels }),
  setTimeWindow: (window) => set({ timeWindow: window }),
  setIsRunning: (isRunning) => set({ isRunning }),
}));
