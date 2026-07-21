import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Lightbulb, Activity, BrainCircuit } from 'lucide-react';
import { 
  GlassCard, 
  SkeletonShimmer, 
  EmptyState, 
  ErrorState, 
  ShapBarChart 
} from '../../../shared/components';
import { pageTransition, slideUp, staggerChildren, fadeIn } from '../../../shared/lib/motion-presets';
import type { ModelOutput } from '@neuroaegis/model-contracts';

export const ExplainabilityPage: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<'random_forest' | 'xgboost' | 'lightgbm'>('random_forest');
  const [data, setData] = useState<ModelOutput | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState(false);

  const fetchExplanation = async () => {
    setIsLoading(true);
    setIsError(false);
    setData(null);

    try {
      const res = await fetch('/api/v1/jobs/latest');
      if (!res.ok) throw new Error('Failed to fetch latest job');
      const latestJob = await res.json();
      
      if (latestJob.status !== 'Completed' || !latestJob.prediction) {
        throw new Error('Latest job not completed or missing prediction data');
      }

      const responseData: ModelOutput = {
        modelName: latestJob.modelName,
        prediction: latestJob.prediction,
        confidence: latestJob.confidence,
        explanation: latestJob.explanation,
        generatedAt: latestJob.generatedAt
      };
      
      setData(responseData);
    } catch (err) {
      console.error(err);
      setIsError(true);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-fetch on mount
  useEffect(() => {
    fetchExplanation();
  }, []);

  const chartFeatures = data?.explanation?.features.map(f => ({
    name: f.featureName,
    value: f.value
  })) || [];

  return (
    <motion.div {...pageTransition} className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Lightbulb className="w-8 h-8 text-[var(--accent-highlight)]" strokeWidth={1.5} />
          <h1 className="text-2xl font-[var(--font-display)] text-[var(--text-primary)]">Model Explainability (SHAP)</h1>
        </div>
        
        <div className="flex gap-3">
          <select 
            value={selectedModel}
            onChange={e => {
              setSelectedModel(e.target.value as any);
              fetchExplanation();
            }}
            className="bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-secondary)]"
          >
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
            <option value="lightgbm">LightGBM</option>
          </select>
          <button 
            onClick={fetchExplanation}
            disabled={isLoading}
            className="px-4 py-2 bg-[var(--accent-primary)] text-[var(--bg-1)] rounded-lg font-[var(--font-body)] text-sm disabled:opacity-50 hover:opacity-90 transition-opacity"
          >
            {isLoading ? 'Analyzing...' : 'Refresh'}
          </button>
        </div>
      </div>

      <motion.div {...staggerChildren} className="flex flex-col gap-6">
        {isLoading && (
          <motion.div {...slideUp}>
            <GlassCard className="p-8 h-96 flex flex-col items-center justify-center gap-4">
               <BrainCircuit className="w-12 h-12 text-[var(--accent-primary)] animate-pulse" />
               <p className="text-[var(--text-secondary)] font-[var(--font-body)]">Computing Shapley values...</p>
            </GlassCard>
          </motion.div>
        )}

        {isError && !isLoading && (
          <motion.div {...slideUp}>
            <ErrorState 
              title="Explanation Failed" 
              message="Unable to fetch SHAP explanations for the selected model." 
              onRetry={fetchExplanation} 
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
              action={{ label: 'Fetch Explanation', onClick: fetchExplanation }} 
            />
          </motion.div>
        )}

        {!isLoading && !isError && data && (
          <>
            <motion.div {...slideUp} className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <GlassCard className="p-6 flex items-center justify-between">
                <div>
                  <div className="text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-1">Base Value</div>
                  <div className="text-3xl font-[var(--font-mono)] text-[var(--text-primary)]">
                    {data.explanation.baseValue.toFixed(4)}
                  </div>
                  <p className="text-xs text-[var(--text-secondary)] mt-2">Average prediction output before features</p>
                </div>
                <div className="w-16 h-16 rounded-full bg-[var(--bg-2)] flex items-center justify-center border border-[var(--bg-3)]">
                  <Activity className="w-8 h-8 text-[var(--accent-highlight)]" />
                </div>
              </GlassCard>

              <GlassCard className="p-6 flex items-center justify-between">
                <div>
                  <div className="text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-1">Top Features Analyzed</div>
                  <div className="text-3xl font-[var(--font-mono)] text-[var(--text-primary)]">
                    {data.explanation.features.length}
                  </div>
                  <p className="text-xs text-[var(--text-secondary)] mt-2">Features shifting the prediction</p>
                </div>
                <div className="w-16 h-16 rounded-full bg-[var(--bg-2)] flex items-center justify-center border border-[var(--bg-3)]">
                  <BrainCircuit className="w-8 h-8 text-[var(--accent-primary)]" />
                </div>
              </GlassCard>
            </motion.div>

            <motion.div {...slideUp}>
              <GlassCard title="Feature Contributions (Force Plot)" className="p-6 h-[500px] flex flex-col">
                <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                   <ShapBarChart features={chartFeatures} maxBars={10} className="mt-4" />
                </div>
                
                <div className="mt-6 pt-4 border-t border-[var(--bg-3)] flex justify-between text-xs font-[var(--font-mono)] text-[var(--text-secondary)]">
                   <div className="flex items-center gap-2">
                     <span className="w-3 h-3 rounded-full bg-[var(--state-danger)]" />
                     Pushes towards Seizure
                   </div>
                   <div className="flex items-center gap-2">
                     <span className="w-3 h-3 rounded-full bg-[var(--accent-primary)]" />
                     Pushes towards Non-Seizure
                   </div>
                </div>
              </GlassCard>
            </motion.div>

            <motion.div {...slideUp}>
              <GlassCard title="Understanding SHAP Values" className="p-6 bg-[var(--bg-2)]/50">
                <p className="font-[var(--font-body)] text-[var(--text-secondary)] text-sm leading-relaxed">
                  SHAP (SHapley Additive exPlanations) values show how much each feature contributes to pushing the model output from the base value (the average model output) to the actual prediction. Features pushing left (red) increase the likelihood of a seizure prediction, while features pushing right (blue/purple) decrease it.
                </p>
              </GlassCard>
            </motion.div>
          </>
        )}
      </motion.div>
    </motion.div>
  );
};

