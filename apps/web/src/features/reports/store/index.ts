import { create } from 'zustand';

interface ReportsState {
  selectedDateRange: { start: string; end: string } | null;
  reportType: 'clinical' | 'technical';
  setSelectedDateRange: (range: { start: string; end: string } | null) => void;
  setReportType: (type: 'clinical' | 'technical') => void;
}

export const useReportsStore = create<ReportsState>((set) => ({
  selectedDateRange: null,
  reportType: 'clinical',
  setSelectedDateRange: (range) => set({ selectedDateRange: range }),
  setReportType: (type) => set({ reportType: type }),
}));
