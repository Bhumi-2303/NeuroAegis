import { useState } from 'react';
import { motion } from 'framer-motion';
import { Brain } from 'lucide-react';
import { 
  GlassCard, 
  SkeletonShimmer, 
  EmptyState, 
  ErrorState, 
  ConfidenceGauge, 
  StatusBadge,
  HudCornerFrame,
  ModelSelectorSegmented,
  InspectorPopoutCard,
  ClinicalMetaTable,
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
  const [showPopout, setShowPopout] = useState(false);
  
  const input = createMockInput(selectedModel);
  const { data, isLoading, isError, refetch } = usePrediction(input);

  const handleModelChange = (model: string) => {
    setSelectedModel(model as 'random_forest' | 'xgboost' | 'lightgbm');
  };

  return (
    <motion.div {...pageTransition} className="p-6 space-y-6">
      <HudCornerFrame label="Seizure Prediction" icon={<Brain size={14} strokeWidth={1.5} />}>
        <h1 className="text-2xl font-display text-[var(--text-primary)]">Seizure Prediction</h1>
      </HudCornerFrame>

      {/* Model Selector */}
      <div className="flex items-center gap-4 flex-wrap">
        <ModelSelectorSegmented
          models={['random_forest', 'xgboost', 'lightgbm'] as const}
          selected={selectedModel}
          onSelect={handleModelChange}
        />
        <button 
          onClick={() => refetch()}
          className="px-4 py-2 glass-card text-[var(--accent-secondary)] hover:text-[var(--accent-highlight)] border-[var(--accent-secondary)] transition-all font-body text-sm"
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
            {/* Confidence Gauge with Inspector Popout */}
            <motion.div {...slideUp} className="lg:col-span-2 relative">
              <GlassCard className="flex flex-col items-center justify-center p-8 h-full min-h-[300px]">
                <HudCornerFrame label="Model Confidence">
                  <div
                    className="cursor-pointer"
                    onClick={() => setShowPopout(!showPopout)}
                  >
                    <ConfidenceGauge 
                      value={data.confidence.value * 100} 
                      size={180} 
                      label={`Band: ${data.confidence.band.toUpperCase()}`} 
                    />
                  </div>
                </HudCornerFrame>
                <div className="mt-8">
                  <StatusBadge 
                    status={data.prediction.label === 'seizure' ? 'critical' : 'normal'} 
                    label={data.prediction.label === 'seizure' ? 'Seizure Detected' : 'Normal Activity'} 
                  />
                </div>
              </GlassCard>

              {/* Inspector Popout */}
              <InspectorPopoutCard
                visible={showPopout}
                liveDot
                className="-right-4 -top-4 w-64"
              >
                <ClinicalMetaTable
                  rows={[
                    { attribute: 'Session ID', value: data.prediction.sessionId },
                    { attribute: 'Model', value: selectedModel.replace('_', ' ') },
                    { attribute: 'Label', value: data.prediction.label },
                    { attribute: 'Confidence', value: `${(data.confidence.value * 100).toFixed(1)}%` },
                    { attribute: 'Timestamp', value: new Date(data.prediction.timestamp).toLocaleTimeString() },
                  ]}
                />
              </InspectorPopoutCard>
            </motion.div>

            {/* Probability Breakdown */}
            <motion.div {...slideUp} className="lg:col-span-1">
              <GlassCard className="h-full p-6 flex flex-col gap-6">
                <HudCornerFrame label="Probability Breakdown">
                  <motion.div {...fadeIn} className="mt-2">
                    <div className="flex justify-between mb-2">
                      <span className="font-body text-[var(--text-secondary)]">Seizure</span>
                      <span className="font-mono text-[var(--text-primary)]">
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

                  <motion.div {...fadeIn} className="mt-4">
                    <div className="flex justify-between mb-2">
                      <span className="font-body text-[var(--text-secondary)]">Non-Seizure</span>
                      <span className="font-mono text-[var(--text-primary)]">
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
                </HudCornerFrame>
              </GlassCard>
            </motion.div>
          </>
        )}
      </motion.div>
    </motion.div>
  );
};
