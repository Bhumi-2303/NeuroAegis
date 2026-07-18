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
  { name: 'gamma', color: '#FF4D6D', label: 'Gamma' },
  { name: 'beta', color: '#FFB020', label: 'Beta' },
  { name: 'alpha', color: '#00FFA3', label: 'Alpha' },
  { name: 'theta', color: '#4B7DFF', label: 'Theta' },
  { name: 'delta', color: '#8B5CF6', label: 'Delta' }
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
    if (isLoading) {
      return (
        <motion.div variants={staggerChildren} className="flex flex-col gap-4">
          <motion.div variants={slideUp}>
            <SkeletonShimmer className="h-64 w-full rounded-xl" />
          </motion.div>
          <motion.div variants={slideUp} className="grid grid-cols-5 gap-4">
            {BANDS.map(b => (
              <SkeletonShimmer key={b.name} className="h-24 w-full rounded-xl" />
            ))}
          </motion.div>
        </motion.div>
      );
    }

    if (error) {
      return (
        <ErrorState 
          title="Frequency Analysis Error" 
          message={error.message || 'Could not load frequency band data.'} 
        />
      );
    }

    if (!chartData || chartData.length === 0) {
      return (
        <EmptyState 
          icon={Radio} 
          title="No Frequency Data" 
          description="Waiting for the frequency analysis streams to initialize." 
        />
      );
    }

    return (
      <motion.div variants={staggerChildren} className="space-y-6">
        <motion.div variants={slideUp}>
          <GlassCard className="p-6 h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  {BANDS.map(({ name, color }) => (
                    <linearGradient key={`gradient-${name}`} id={`gradient-${name}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={color} stopOpacity={0} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis 
                  dataKey="timestamp" 
                  stroke="var(--text-secondary)" 
                  fontSize={12} 
                  tickFormatter={(val) => new Date(val).toLocaleTimeString([], { hour12: false, second: '2-digit' })}
                  tickMargin={10}
                />
                <YAxis 
                  stroke="var(--text-secondary)" 
                  fontSize={12} 
                  tickMargin={10}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'var(--bg-2)', 
                    borderColor: 'rgba(255,255,255,0.1)', 
                    borderRadius: '8px',
                    color: 'var(--text-primary)',
                    fontFamily: 'var(--font-mono)'
                  }} 
                />
                {BANDS.map(({ name, color }) => (
                  <Area
                    key={name}
                    type="monotone"
                    dataKey={name}
                    stroke={color}
                    fillOpacity={1}
                    fill={`url(#gradient-${name})`}
                    strokeWidth={2}
                    style={{ filter: `drop-shadow(0 0 4px ${color}80)` }} // Glowing stroke
                    isAnimationActive={false}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </GlassCard>
        </motion.div>

        {/* Bottom Stat Tiles */}
        {latestAverages && (
          <motion.div variants={staggerChildren} className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {BANDS.map(({ name, label, color }) => (
              <motion.div key={name} variants={slideUp}>
                <GlassCard className="p-4 flex flex-col items-center justify-center text-center">
                  <span className="text-sm font-body" style={{ color: color }}>{label}</span>
                  <span className="text-xl font-mono mt-2" style={{ color: 'var(--text-primary)' }}>
                    {latestAverages[name].toFixed(2)} µV²
                  </span>
                </GlassCard>
              </motion.div>
            ))}
          </motion.div>
        )}
      </motion.div>
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
