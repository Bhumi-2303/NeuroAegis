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
  ErrorState,
  HudCornerFrame,
  FrequencyBandRow,
} from '../../../shared/components';
import { pageTransition, staggerChildren, slideUp } from '../../../shared/lib/motion-presets';

import { useFrequencyBands } from '../hooks/useFrequencyBands';

const BANDS = [
  { name: 'gamma', color: '#FF4D6D', label: 'Gamma', greek: 'γ', hz: '30–100 Hz' },
  { name: 'beta', color: '#FFB020', label: 'Beta', greek: 'β', hz: '13–30 Hz' },
  { name: 'alpha', color: '#00FFA3', label: 'Alpha', greek: 'α', hz: '8–13 Hz' },
  { name: 'theta', color: '#4B7DFF', label: 'Theta', greek: 'θ', hz: '4–8 Hz' },
  { name: 'delta', color: '#8B5CF6', label: 'Delta', greek: 'δ', hz: '0.5–4 Hz' }
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

  const bandStats = useMemo(() => {
    if (chartData.length === 0) return null;
    
    const result: Record<string, { avg: number; sparkline: number[] }> = {};
    BANDS.forEach(({ name }) => {
      const values = chartData.map(d => d[name]).filter(v => v !== undefined && v !== null);
      if (values.length > 0) {
        result[name] = {
          avg: values.reduce((sum: number, val: number) => sum + val, 0) / values.length,
          sparkline: values.slice(-20),
        };
      } else {
        result[name] = { avg: 0, sparkline: [] };
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
                    style={{ filter: `drop-shadow(0 0 4px ${color}80)` }}
                    isAnimationActive={false}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </GlassCard>
        </motion.div>

        {/* FrequencyBandRow list — replaces the old stat tiles */}
        {bandStats && (
          <motion.div variants={slideUp}>
            <GlassCard className="p-4">
              <HudCornerFrame label="Band Power Levels">
                <div className="flex flex-col gap-0.5 mt-2">
                  {BANDS.map(({ name, label, color, greek, hz }) => (
                    <FrequencyBandRow
                      key={name}
                      bandName={label}
                      greekLetter={greek}
                      color={color}
                      hzRange={hz}
                      sparklineData={bandStats[name].sparkline}
                      avgPower={bandStats[name].avg}
                    />
                  ))}
                </div>
              </HudCornerFrame>
            </GlassCard>
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
      <header className="mb-6">
        <HudCornerFrame label="Frequency Band Analysis" icon={<Radio size={14} strokeWidth={1.5} />}>
          <h1 className="text-2xl font-display text-[var(--text-primary)]">Frequency Band Analysis</h1>
        </HudCornerFrame>
      </header>

      <main className="flex-1">
        {renderContent()}
      </main>
    </motion.div>
  );
};

export default FrequencyAnalysisPage;
