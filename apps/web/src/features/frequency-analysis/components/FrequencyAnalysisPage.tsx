import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Radio } from 'lucide-react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';

import { 
  GlassCard, 
  SkeletonShimmer, 
  EmptyState, 
  ErrorState 
} from '../../../shared/components';
import { pageTransition, staggerChildren, slideUp } from '../../../shared/lib/motion-presets';

import { useFrequencyBands } from '../hooks/useFrequencyBands';

const BANDS = [
  { name: 'gamma', color: '#F87171', label: 'Gamma' },
  { name: 'beta', color: '#FBBF24', label: 'Beta' },
  { name: 'alpha', color: '#34D399', label: 'Alpha' },
  { name: 'theta', color: '#60A5FA', label: 'Theta' },
  { name: 'delta', color: '#A78BFA', label: 'Delta' }
] as const;

export const FrequencyAnalysisPage: React.FC = () => {
  const { data, error, isLoading } = useFrequencyBands();

  const chartData = useMemo(() => {
    if (!data) return [];
    
    const groupedData = new Map<string, any>();
    
    BANDS.forEach(({ name }) => {
      const bandData = data[name as keyof typeof data];
      if (Array.isArray(bandData)) {
        bandData.forEach((point) => {
          const entry = groupedData.get(point.timestamp) || { timestamp: point.timestamp };
          entry[name] = point.value;
          groupedData.set(point.timestamp, entry);
        });
      }
    });
    
    return Array.from(groupedData.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }, [data]);

  const latestAverages = useMemo(() => {
    if (chartData.length === 0) return null;
    
    const result: Record<string, number> = {};
    BANDS.forEach(({ name }) => {
      // simple average of last N points or overall average for display
      const values = chartData.map(d => d[name]).filter(v => v !== undefined && v !== null);
      if (values.length > 0) {
        result[name] = values.reduce((sum, val) => sum + val, 0) / values.length;
      } else {
        result[name] = 0;
      }
    });
    return result;
  }, [chartData]);

  const renderContent = () => {
    return (
      <div className="flex flex-col items-center justify-center h-[500px] text-center border-2 border-dashed border-[var(--bg-4)] rounded-xl bg-[var(--bg-2)]/50">
        <Lock className="text-gray-500 mb-3" size={32} />
        <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2">Awaiting Backend Integration</h3>
        <p className="text-sm text-[var(--text-secondary)] max-w-[400px]">
          Frequency band analysis and spectral power streams will be available when live EEG processing is implemented on the backend.
        </p>
      </div>
    );
  };

  return (
    <motion.div 
      className="p-6 h-full flex flex-col"
      variants={pageTransition}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      <header className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-[var(--bg-2)] border border-[rgba(255,255,255,0.1)] rounded-lg">
          <Radio className="w-6 h-6 text-[var(--accent-primary)]" />
        </div>
        <h1 className="text-2xl font-display text-[var(--text-primary)]">Frequency Band Analysis</h1>
      </header>

      <main className="flex-1">
        {renderContent()}
      </main>
    </motion.div>
  );
};

export default FrequencyAnalysisPage;
