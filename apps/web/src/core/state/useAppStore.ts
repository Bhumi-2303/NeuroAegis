import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { ModelOutput } from '@neuroaegis/model-contracts';

interface AppState {
  activeModelId: string;
  setActiveModelId: (id: string) => void;
  
  isSidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
  
  activePatientId: string | null;
  setActivePatientId: (id: string | null) => void;

  latestPrediction: ModelOutput | null;
  setLatestPrediction: (data: ModelOutput | null) => void;
}

export const useAppStore = create<AppState>()(
  devtools(
    (set) => ({
      activeModelId: 'rf-01', // Default model
      setActiveModelId: (id) => set({ activeModelId: id }),
      
      isSidebarCollapsed: false,
      setSidebarCollapsed: (collapsed) => set({ isSidebarCollapsed: collapsed }),
      
      activePatientId: null,
      setActivePatientId: (id) => set({ activePatientId: id }),

      latestPrediction: null,
      setLatestPrediction: (data) => set({ latestPrediction: data }),
    }),
    { name: 'AppStore' }
  )
);
