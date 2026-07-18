import { create } from 'zustand';

interface PatientsState {
  selectedPatientId: string | null;
  searchQuery: string;
  setSelectedPatientId: (id: string | null) => void;
  setSearchQuery: (query: string) => void;
}

export const usePatientsStore = create<PatientsState>((set) => ({
  selectedPatientId: null,
  searchQuery: '',
  setSelectedPatientId: (id) => set({ selectedPatientId: id }),
  setSearchQuery: (query) => set({ searchQuery: query }),
}));
