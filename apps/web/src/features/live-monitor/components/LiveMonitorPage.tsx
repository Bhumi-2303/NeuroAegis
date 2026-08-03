import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, FastForward, MonitorPlay, AlertTriangle } from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

import { GlassCard, SkeletonShimmer, ErrorState } from '../../../shared/components';
import { pageTransition, staggerChildren, slideUp } from '../../../shared/lib/motion-presets';

interface WindowData {
  window_idx: number;
  target: number;
  features: Record<string, number>;
}

interface LiveMonitorData {
  record: string;
  windows: WindowData[];
}

interface ChartPoint {
  time: number;
  probability: number;
  featureValue: number;
  isSeizure: boolean;
  modelDetected: boolean;
}

export const LiveMonitorPage: React.FC = () => {
  const [data, setData] = useState<LiveMonitorData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [currentIndex, setCurrentIndex] = useState(0);
  
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [isAlertActive, setIsAlertActive] = useState(false);
  
  // Track interval
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch initial data
  useEffect(() => {
    const fetchMonitorData = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v2/demo/live-monitor-data`);
        if (!response.ok) throw new Error('Failed to fetch live monitor data');
        const json = await response.json();
        setData(json);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    fetchMonitorData();
  }, []);

  const processNextWindow = useCallback(async () => {
    if (!data || currentIndex >= data.windows.length) {
      setIsPlaying(false);
      return;
    }

    const currentWindow = data.windows[currentIndex];
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v2/demo/predict-features`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: currentWindow.features })
      });
      
      if (!response.ok) throw new Error('Prediction failed');
      const result = await response.json();
      
      const probability = result.probability_seizure * 100;
      const modelDetected = probability > 50;
      const isGenuineSeizure = currentWindow.target === 1;
      
      const newPoint: ChartPoint = {
        time: currentWindow.window_idx * 2, // Assuming 2s windows
        probability,
        featureValue: currentWindow.features['Ch0_energy'] || 0, // Top feature
        isSeizure: isGenuineSeizure,
        modelDetected
      };
      
      setChartData(prev => {
        const newData = [...prev, newPoint];
        // Keep last 30 points (60 seconds) for rolling window
        if (newData.length > 30) return newData.slice(newData.length - 30);
        return newData;
      });
      
      setIsAlertActive(modelDetected && isGenuineSeizure);
      setCurrentIndex(prev => prev + 1);
      
    } catch (err) {
      console.error('Error during live prediction:', err);
      setIsPlaying(false);
    }
  }, [data, currentIndex]);

  useEffect(() => {
    if (isPlaying) {
      // Base speed: 1000ms per window (if 1x). Accelerated by speed multiplier.
      const intervalTime = 1000 / speed;
      timerRef.current = setInterval(processNextWindow, intervalTime);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, speed, processNextWindow]);

  const togglePlay = () => setIsPlaying(!isPlaying);
  
  const cycleSpeed = () => {
    setSpeed(prev => prev === 1 ? 5 : prev === 5 ? 10 : 1);
  };

  if (isLoading) {
    return (
      <div className="p-6 h-full space-y-4">
        <SkeletonShimmer className="h-48 w-full rounded-xl" />
        <SkeletonShimmer className="h-48 w-full rounded-xl" />
      </div>
    );
  }

  if (error) {
    return <ErrorState title="Live Monitor Error" message={error} />;
  }

  return (
    <motion.div 
      className="p-6 h-full flex flex-col relative"
      variants={pageTransition}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      <AnimatePresence>
        {isAlertActive && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="absolute top-6 left-1/2 -translate-x-1/2 z-50 bg-red-500/90 backdrop-blur-md text-white px-8 py-3 rounded-full flex items-center gap-3 shadow-xl border border-red-400"
          >
            <AlertTriangle className="animate-pulse" />
            <span className="font-bold tracking-wider">SEIZURE DETECTED</span>
            <span className="text-red-100 text-sm ml-2">Ground Truth Confirmed</span>
          </motion.div>
        )}
      </AnimatePresence>

      <header className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[var(--bg-2)] border border-[rgba(255,255,255,0.1)] rounded-lg">
            <MonitorPlay className="w-6 h-6 text-[var(--accent-primary)]" />
          </div>
          <div>
            <h1 className="text-2xl font-display text-[var(--text-primary)]">Live Monitor</h1>
            {data && <p className="text-sm text-[var(--text-secondary)]">Source: {data.record} • Accelerated Playback</p>}
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <button 
            onClick={cycleSpeed}
            className="flex items-center gap-2 px-4 py-2 bg-[var(--bg-2)] border border-[rgba(255,255,255,0.1)] rounded-lg text-[var(--text-primary)] hover:bg-[var(--bg-3)] transition-colors"
          >
            <FastForward size={16} />
            <span>{speed}x</span>
          </button>
          
          <button 
            onClick={togglePlay}
            className={`flex items-center gap-2 px-6 py-2 rounded-lg font-medium transition-colors ${
              isPlaying 
                ? 'bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500/20' 
                : 'bg-[var(--accent-primary)] text-white hover:bg-[var(--accent-primary-hover)]'
            }`}
          >
            {isPlaying ? (
              <><Pause size={18} /> Pause</>
            ) : (
              <><Play size={18} /> Start Stream</>
            )}
          </button>
        </div>
      </header>

      <main className="flex-1 space-y-6">
        <motion.div variants={staggerChildren} className="grid grid-cols-1 gap-6 h-full">
          
          {/* Chart 1: Model Probability */}
          <motion.div variants={slideUp} className="h-1/2">
            <GlassCard className="p-6 h-full flex flex-col">
              <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4 uppercase tracking-wider">Model Confidence over time</h3>
              <div className="flex-1">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                    <XAxis dataKey="time" stroke="var(--text-secondary)" fontSize={12} tickFormatter={(val) => `${val}s`} />
                    <YAxis stroke="var(--text-secondary)" fontSize={12} domain={[0, 100]} tickFormatter={(val) => `${val}%`} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'var(--bg-2)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }} 
                      formatter={(val: number) => [`${val.toFixed(1)}%`, 'Probability']}
                      labelFormatter={(label) => `Time: ${label}s`}
                    />
                    <ReferenceLine y={50} stroke="rgba(255,255,255,0.2)" strokeDasharray="3 3" />
                    <Line 
                      type="monotone" 
                      dataKey="probability" 
                      stroke={isAlertActive ? '#ef4444' : 'var(--accent-primary)'} 
                      strokeWidth={3}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>
          </motion.div>

          {/* Chart 2: Top Feature */}
          <motion.div variants={slideUp} className="h-1/2">
            <GlassCard className="p-6 h-full flex flex-col">
              <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4 uppercase tracking-wider">Ch0_energy over time</h3>
              <div className="flex-1">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                    <XAxis dataKey="time" stroke="var(--text-secondary)" fontSize={12} tickFormatter={(val) => `${val}s`} />
                    <YAxis stroke="var(--text-secondary)" fontSize={12} domain={['auto', 'auto']} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'var(--bg-2)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }} 
                      formatter={(val: number) => [val.toExponential(2), 'Energy']}
                      labelFormatter={(label) => `Time: ${label}s`}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="featureValue" 
                      stroke="var(--accent-secondary)" 
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>
          </motion.div>
          
        </motion.div>
      </main>
    </motion.div>
  );
};
