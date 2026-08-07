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
  EmptyState, 
  ErrorState 
} from '../../../shared/components';
import { pageTransition } from '../../../shared/lib/motion-presets';

import { useEEGStream } from '../hooks/useEEGStream';
import { useEegStore } from '../store';

const CHANNEL_COLORS: Record<string, string> = {
  Fp1: '#14B8A6',
  Fp2: '#60A5FA',
  C3: '#A78BFA',
  C4: '#34D399',
  O1: '#FBBF24',
  O2: '#F87171',
  Default: '#94A3B8'
};

export const EEGMonitorPage: React.FC = () => {
  const { selectedChannels, timeWindow: _timeWindow, isRunning } = useEegStore();
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
    if (error) {
      return (
        <ErrorState 
          title="Stream Connection Error" 
          message={error.message || "Failed to connect to the live EEG stream."}
        />
      );
    }

    if (isLoading && (!chartData || chartData.length === 0)) {
      return (
        <GlassCard className="h-[500px] flex items-center justify-center">
          <div className="flex flex-col items-center">
            <div className="w-8 h-8 border-4 border-[var(--accent-primary)] border-t-transparent rounded-full animate-spin mb-4" />
            <p className="text-[var(--text-secondary)]">Connecting to stream...</p>
          </div>
        </GlassCard>
      );
    }

    if (!chartData || chartData.length === 0) {
      return <EmptyState title="Waiting for signal data..." />;
    }

    return (
      <GlassCard className="h-[500px] p-6 flex flex-col">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-lg font-semibold text-[var(--text-primary)]">Real-time Acquisition</h2>
          <div className="flex items-center gap-2 text-xs">
            <span className="relative flex h-3 w-3">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isRunning ? 'bg-[var(--state-success)]' : 'bg-[var(--text-muted)]'}`}></span>
              <span className={`relative inline-flex rounded-full h-3 w-3 ${isRunning ? 'bg-[var(--state-success)]' : 'bg-[var(--text-muted)]'}`}></span>
            </span>
            <span className="text-[var(--text-secondary)] font-medium">
              {isRunning ? 'Streaming active' : 'Stream paused'}
            </span>
          </div>
        </div>
        <div className="flex-1 w-full relative">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(tick) => {
                  const date = new Date(tick);
                  return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}:${date.getSeconds().toString().padStart(2, '0')}.${Math.floor(date.getMilliseconds() / 100)}`;
                }}
                stroke="#6B7280"
                fontSize={12}
                minTickGap={50}
              />
              <YAxis 
                stroke="#6B7280"
                fontSize={12}
                domain={['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--bg-2)',
                  borderColor: 'var(--bg-4)',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)',
                  color: 'var(--text-primary)'
                }}
                labelFormatter={(label) => new Date(label).toISOString().split('T')[1].replace('Z', '')}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              {selectedChannels.map(channel => (
                <Line
                  key={channel}
                  type="monotone"
                  dataKey={channel}
                  stroke={CHANNEL_COLORS[channel] || CHANNEL_COLORS.Default}
                  dot={false}
                  isAnimationActive={false}
                  strokeWidth={1.5}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </GlassCard>
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
