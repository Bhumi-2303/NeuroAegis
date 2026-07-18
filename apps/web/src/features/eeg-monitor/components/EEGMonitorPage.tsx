import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Activity } from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend
} from 'recharts';

import { 
  GlassCard, 
  SkeletonShimmer, 
  EmptyState, 
  ErrorState 
} from '../../../shared/components';
import { pageTransition, staggerChildren, slideUp } from '../../../shared/lib/motion-presets';

import { useEEGStream } from '../hooks/useEEGStream';
import { useEegStore } from '../store';

const CHANNEL_COLORS: Record<string, string> = {
  Fp1: '#00E5FF',
  Fp2: '#4B7DFF',
  C3: '#8B5CF6',
  C4: '#00FFA3',
  O1: '#FFB020',
  O2: '#FF4D6D',
  Default: '#94A3B8'
};

export const EEGMonitorPage: React.FC = () => {
  const { selectedChannels, timeWindow, isRunning } = useEegStore();
  const { data, error, isLoading } = useEEGStream(selectedChannels);

  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    // Group data points by timestamp so Recharts can plot multiple lines correctly
    const groupedData = new Map<string, any>();
    
    data.forEach((point) => {
      const entry = groupedData.get(point.timestamp) || { timestamp: point.timestamp };
      if (point.channel) {
        entry[point.channel] = point.value;
      }
      groupedData.set(point.timestamp, entry);
    });
    
    return Array.from(groupedData.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }, [data]);

  const renderContent = () => {
    if (isLoading) {
      return (
        <motion.div variants={staggerChildren} className="flex flex-col gap-4">
          <motion.div variants={slideUp}>
            <SkeletonShimmer className="h-48 w-full rounded-xl" />
          </motion.div>
          <motion.div variants={slideUp}>
            <SkeletonShimmer className="h-48 w-full rounded-xl" />
          </motion.div>
          <motion.div variants={slideUp}>
            <SkeletonShimmer className="h-48 w-full rounded-xl" />
          </motion.div>
          <motion.div variants={slideUp}>
            <SkeletonShimmer className="h-48 w-full rounded-xl" />
          </motion.div>
        </motion.div>
      );
    }

    if (error) {
      return (
        <ErrorState 
          title="Failed to Load EEG Stream" 
          message={error.message || 'An unexpected error occurred while fetching EEG data.'} 
        />
      );
    }

    if (!chartData || chartData.length === 0) {
      return (
        <EmptyState 
          icon={Activity} 
          title="No EEG data available" 
          description="Waiting for the EEG data stream to initialize or connect." 
        />
      );
    }

    return (
      <motion.div variants={staggerChildren} className="space-y-6">
        <motion.div variants={slideUp}>
          <GlassCard className="p-6 h-[500px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
                <Legend 
                  verticalAlign="top" 
                  height={36} 
                  iconType="circle"
                  formatter={(value) => <span className="text-[var(--text-primary)] font-mono ml-1">{value}</span>}
                />
                {selectedChannels.map((channel) => (
                  <Line 
                    key={channel}
                    type="monotone" 
                    dataKey={channel} 
                    stroke={CHANNEL_COLORS[channel] || CHANNEL_COLORS.Default} 
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </GlassCard>
        </motion.div>

        {/* Bottom Stat Cards */}
        <motion.div variants={staggerChildren} className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <motion.div variants={slideUp}>
            <GlassCard className="p-4 flex flex-col items-center justify-center text-center">
              <span className="text-sm text-[var(--text-secondary)] font-body">Active Channels</span>
              <span className="text-2xl text-[var(--accent-primary)] font-mono mt-1">{selectedChannels.length}</span>
            </GlassCard>
          </motion.div>
          <motion.div variants={slideUp}>
            <GlassCard className="p-4 flex flex-col items-center justify-center text-center">
              <span className="text-sm text-[var(--text-secondary)] font-body">Time Window</span>
              <span className="text-2xl text-[var(--accent-secondary)] font-mono mt-1">{timeWindow}s</span>
            </GlassCard>
          </motion.div>
          <motion.div variants={slideUp}>
            <GlassCard className="p-4 flex flex-col items-center justify-center text-center">
              <span className="text-sm text-[var(--text-secondary)] font-body">Status</span>
              <span className="text-2xl text-[var(--state-success)] font-mono mt-1">
                {isRunning ? 'RUNNING' : 'PAUSED'}
              </span>
            </GlassCard>
          </motion.div>
        </motion.div>
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
          <Activity className="w-6 h-6 text-[var(--accent-primary)]" />
        </div>
        <h1 className="text-2xl font-display text-[var(--text-primary)]">Neural Activity Monitor</h1>
      </header>

      <main className="flex-1">
        {renderContent()}
      </main>
    </motion.div>
  );
};

export default EEGMonitorPage;
