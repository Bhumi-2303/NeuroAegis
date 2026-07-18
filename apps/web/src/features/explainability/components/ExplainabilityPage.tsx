import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Lightbulb } from 'lucide-react';
import { 
  GlassCard, 
  SkeletonShimmer, 
  EmptyState, 
  ErrorState, 
  ShapBarChart 
} from '../../../shared/components';
import { pageTransition, slideUp, staggerChildren } from '../../../shared/lib/motion-presets';
import { useExplanation } from '../hooks/useExplanation';
import type { ModelInput } from '@neuroaegis/model-contracts';

const createMockInput = (modelName: string): ModelInput => ({
  sessionId: 'mock-session-456',
  signalWindow: [0.1, -0.2, 0.4],
  channelIds: ['CH1', 'CH2'],
  samplingRateHz: 256,
  timestamp: new Date().toISOString(),
  metadata: { modelName }
});

export const ExplainabilityPage: React.FC = () => {
  const [selectedModel] = useState<'random_forest' | 'xgboost' | 'lightgbm'>('random_forest');
  
  const input = createMockInput(selectedModel);
  const { data, isLoading, isError, refetch } = useExplanation(input);

  const chartFeatures = data?.explanation?.features.map(f => ({
    name: f.featureName,
    value: f.value
  })) || [];

  return (
    <motion.div {...pageTransition} className="p-6 space-y-6">
      <div className="flex items-center gap-3 mb-6">
        <Lightbulb className="w-8 h-8 text-[var(--accent-highlight)]" strokeWidth={1.5} />
        <h1 className="text-2xl font-[var(--font-display)] text-[var(--text-primary)]">Model Explainability</h1>
      </div>

      <motion.div {...staggerChildren} className="flex flex-col gap-6">
        {isLoading && (
          <motion.div {...slideUp}>
            <GlassCard className="p-8 h-96">
              <SkeletonShimmer className="w-full h-full rounded-xl" />
            </GlassCard>
          </motion.div>
        )}

        {isError && !isLoading && (
          <motion.div {...slideUp}>
            <ErrorState 
              title="Explanation Failed" 
              message="Unable to fetch SHAP explanations for the selected model." 
              onRetry={() => refetch()} 
              retryLabel="Retry Explanation" 
            />
          </motion.div>
        )}

        {!isLoading && !isError && !data && (
          <motion.div {...slideUp}>
            <EmptyState 
              icon={Lightbulb} 
              title="No Explanation Available" 
              description="No feature attribution data is available for this model." 
              action={{ label: 'Fetch Explanation', onClick: () => refetch() }} 
            />
          </motion.div>
        )}

        {!isLoading && !isError && data && (
          <>
            <motion.div {...slideUp}>
              <GlassCard className="p-6">
                <div className="flex gap-12">
                  <div>
                    <div className="text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-1">Base Value</div>
                    <div className="text-2xl font-[var(--font-mono)] text-[var(--text-primary)]">
                      {data.explanation.baseValue.toFixed(4)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-1">Total Features Analyzed</div>
                    <div className="text-2xl font-[var(--font-mono)] text-[var(--text-primary)]">
                      {data.explanation.features.length}
                    </div>
                  </div>
                </div>
              </GlassCard>
            </motion.div>

            <motion.div {...slideUp}>
              <GlassCard title="Feature Contributions" className="p-6 h-[400px]">
                <ShapBarChart features={chartFeatures} />
              </GlassCard>
            </motion.div>

            <motion.div {...slideUp}>
              <GlassCard title="Understanding SHAP Values" className="p-6">
                <p className="font-[var(--font-body)] text-[var(--text-secondary)] text-sm leading-relaxed">
                  SHAP (SHapley Additive exPlanations) values show how much each feature contributes to pushing the model output from the base value (the average model output) to the actual prediction. Features with positive values push the prediction higher (e.g., towards seizure), while those with negative values push the prediction lower (e.g., towards non-seizure).
                </p>
              </GlassCard>
            </motion.div>
          </>
        )}
      </motion.div>
    </motion.div>
  );
};
