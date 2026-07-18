import { useState } from 'react';
import { motion } from 'framer-motion';
import { Brain } from 'lucide-react';
import { 
  GlassCard, 
  SkeletonShimmer, 
  EmptyState, 
  ErrorState, 
  ConfidenceGauge, 
  StatusBadge 
} from '../../../shared/components';
import { pageTransition, slideUp, staggerChildren, fadeIn } from '../../../shared/lib/motion-presets';
import { usePrediction } from '../hooks/usePrediction';
import type { ModelInput } from '@neuroaegis/model-contracts';

const createMockInput = (modelName: string): ModelInput => ({
  sessionId: 'mock-session-123',
  signalWindow: [0.1, -0.2, 0.3, 0.5, -0.1],
  channelIds: ['EEG-Fpz-Cz', 'EEG-Pz-Oz'],
  samplingRateHz: 256,
  timestamp: new Date().toISOString(),
  metadata: { modelName }
});

export const SeizurePredictionPage: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<'random_forest' | 'xgboost' | 'lightgbm'>('random_forest');
  
  // Store is available for future threshold/autoRefresh usage
  // const { threshold, autoRefresh } = usePredictionStore();
  const input = createMockInput(selectedModel);
  const { data, isLoading, isError, refetch } = usePrediction(input);

  const handleModelChange = (model: 'random_forest' | 'xgboost' | 'lightgbm') => {
    setSelectedModel(model);
  };

  return (
    <motion.div {...pageTransition} className="p-6 space-y-6">
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-8 h-8 text-[var(--accent-primary)]" strokeWidth={1.5} />
        <h1 className="text-2xl font-[var(--font-display)] text-[var(--text-primary)]">Seizure Prediction</h1>
      </div>

      <div className="flex gap-2">
        {(['random_forest', 'xgboost', 'lightgbm'] as const).map(model => (
          <button
            key={model}
            onClick={() => handleModelChange(model)}
            className={`px-4 py-2 rounded-lg font-[var(--font-body)] text-sm transition-all ${
              selectedModel === model 
                ? 'bg-[var(--accent-primary)] text-[var(--bg-1)] font-medium' 
                : 'glass-card text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            {model.replace('_', ' ').toUpperCase()}
          </button>
        ))}
        <button 
          onClick={() => refetch()}
          className="ml-auto px-4 py-2 glass-card text-[var(--accent-secondary)] hover:text-[var(--accent-highlight)] border-[var(--accent-secondary)] transition-all font-[var(--font-body)] text-sm"
        >
          Run Prediction
        </button>
      </div>

      <motion.div {...staggerChildren} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {isLoading && (
          <motion.div {...slideUp} className="lg:col-span-3">
            <GlassCard className="p-8 h-64 flex flex-col justify-center items-center">
              <SkeletonShimmer className="w-full h-full rounded-xl" />
            </GlassCard>
          </motion.div>
        )}

        {isError && !isLoading && (
          <motion.div {...slideUp} className="lg:col-span-3">
            <ErrorState 
              title="Prediction Failed" 
              message="Unable to generate prediction from the selected model." 
              onRetry={() => refetch()} 
              retryLabel="Retry Prediction" 
            />
          </motion.div>
        )}

        {!isLoading && !isError && !data && (
          <motion.div {...slideUp} className="lg:col-span-3">
            <EmptyState 
              icon={Brain} 
              title="No Data Available" 
              description="No prediction data is available for this model." 
              action={{ label: 'Run Prediction', onClick: () => refetch() }} 
            />
          </motion.div>
        )}

        {!isLoading && !isError && data && (
          <>
            <motion.div {...slideUp} className="lg:col-span-2">
              <GlassCard title="Model Confidence" className="flex flex-col items-center justify-center p-8 h-full min-h-[300px]">
                <ConfidenceGauge 
                  value={data.confidence.value * 100} 
                  size={180} 
                  label={`Band: ${data.confidence.band.toUpperCase()}`} 
                />
                <div className="mt-8">
                  <StatusBadge 
                    status={data.prediction.label === 'seizure' ? 'critical' : 'normal'} 
                    label={data.prediction.label === 'seizure' ? 'Seizure Detected' : 'Normal Activity'} 
                  />
                </div>
              </GlassCard>
            </motion.div>

            <motion.div {...slideUp} className="lg:col-span-1">
              <GlassCard title="Probability Breakdown" className="h-full p-6 flex flex-col gap-6">
                <motion.div {...fadeIn}>
                  <div className="flex justify-between mb-2">
                    <span className="font-[var(--font-body)] text-[var(--text-secondary)]">Seizure</span>
                    <span className="font-[var(--font-mono)] text-[var(--text-primary)]">
                      {(data.prediction.probabilities.seizure * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-[var(--bg-3)] rounded-full h-2">
                    <div 
                      className="bg-[var(--state-danger)] h-2 rounded-full" 
                      style={{ width: `${data.prediction.probabilities.seizure * 100}%` }}
                    />
                  </div>
                </motion.div>

                <motion.div {...fadeIn}>
                  <div className="flex justify-between mb-2">
                    <span className="font-[var(--font-body)] text-[var(--text-secondary)]">Non-Seizure</span>
                    <span className="font-[var(--font-mono)] text-[var(--text-primary)]">
                      {(data.prediction.probabilities.non_seizure * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-[var(--bg-3)] rounded-full h-2">
                    <div 
                      className="bg-[var(--state-success)] h-2 rounded-full" 
                      style={{ width: `${data.prediction.probabilities.non_seizure * 100}%` }}
                    />
                  </div>
                </motion.div>
              </GlassCard>
            </motion.div>
          </>
        )}
      </motion.div>
    </motion.div>
  );
};
