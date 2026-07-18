import { useState } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, AlertCircle } from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { useEvaluationMetrics } from '../hooks/useEvaluationMetrics';
import { GlassCard, SkeletonShimmer, EmptyState, ErrorState } from '../../../shared/components';
import { pageTransition, fadeIn } from '../../../shared/lib/motion-presets';

export const ReportsPage = () => {
  const [modelName, setModelName] = useState<'random_forest' | 'xgboost' | 'lightgbm'>('random_forest');
  const { data, isLoading, isError, error, refetch } = useEvaluationMetrics(modelName);

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {[...Array(5)].map((_, i) => (
            <GlassCard key={i} className="h-24">
              <SkeletonShimmer className="w-full h-full" />
            </GlassCard>
          ))}
          <GlassCard className="col-span-1 md:col-span-5 h-96">
            <SkeletonShimmer className="w-full h-full" />
          </GlassCard>
          <GlassCard className="col-span-1 md:col-span-5 h-64">
            <SkeletonShimmer className="w-full h-full" />
          </GlassCard>
        </div>
      );
    }

    if (isError) {
      return (
        <ErrorState
          title="Failed to load metrics"
          message={error instanceof Error ? error.message : 'An unknown error occurred'}
          onRetry={refetch}
        />
      );
    }

    if (!data) {
      return (
        <EmptyState
          icon={AlertCircle}
          title="No Data Available"
          description="Evaluation metrics for the selected model are not available."
        />
      );
    }

    const formatPercent = (val: number) => `${(val * 100).toFixed(1)}%`;

    return (
      <motion.div variants={fadeIn} initial="initial" animate="animate" className="space-y-6">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <GlassCard className="flex flex-col items-center justify-center p-4">
            <div className="font-mono text-3xl font-bold text-[var(--accent-primary)]">
              {formatPercent(data.accuracy)}
            </div>
            <div className="font-body text-sm text-[var(--text-secondary)] mt-1">Accuracy</div>
          </GlassCard>
          <GlassCard className="flex flex-col items-center justify-center p-4">
            <div className="font-mono text-3xl font-bold text-[var(--accent-secondary)]">
              {formatPercent(data.precision)}
            </div>
            <div className="font-body text-sm text-[var(--text-secondary)] mt-1">Precision</div>
          </GlassCard>
          <GlassCard className="flex flex-col items-center justify-center p-4">
            <div className="font-mono text-3xl font-bold text-[var(--accent-highlight)]">
              {formatPercent(data.recall)}
            </div>
            <div className="font-body text-sm text-[var(--text-secondary)] mt-1">Recall</div>
          </GlassCard>
          <GlassCard className="flex flex-col items-center justify-center p-4">
            <div className="font-mono text-3xl font-bold text-[var(--state-success)]">
              {formatPercent(data.f1Score)}
            </div>
            <div className="font-body text-sm text-[var(--text-secondary)] mt-1">F1 Score</div>
          </GlassCard>
          <GlassCard className="flex flex-col items-center justify-center p-4">
            <div className="font-mono text-3xl font-bold text-[var(--state-warning)]">
              {formatPercent(data.rocAuc)}
            </div>
            <div className="font-body text-sm text-[var(--text-secondary)] mt-1">ROC-AUC</div>
          </GlassCard>
        </div>

        <GlassCard title="ROC Curve" className="p-4 h-96">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data.rocCurve} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorTpr" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-3)" />
              <XAxis 
                dataKey="fpr" 
                type="number" 
                domain={[0, 1]} 
                tick={{ fill: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}
                stroke="var(--bg-3)"
              />
              <YAxis 
                type="number" 
                domain={[0, 1]}
                tick={{ fill: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}
                stroke="var(--bg-3)"
              />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--bg-1)', border: '1px solid var(--bg-3)', borderRadius: '8px' }}
                itemStyle={{ color: 'var(--text-primary)' }}
              />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--text-secondary)" strokeDasharray="3 3" />
              <Area 
                type="monotone" 
                dataKey="tpr" 
                stroke="var(--accent-primary)" 
                fillOpacity={1} 
                fill="url(#colorTpr)" 
                strokeWidth={2}
                activeDot={{ r: 6, fill: 'var(--accent-highlight)' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </GlassCard>

        <GlassCard title="Confusion Matrix" className="p-4">
          <div className="grid grid-cols-2 gap-4">
            <GlassCard className="flex flex-col items-center justify-center p-6 border border-[var(--state-success)]/30 bg-[var(--state-success)]/5">
              <div className="font-mono text-4xl font-bold text-[var(--state-success)]">
                {data.confusionMatrix.truePositive}
              </div>
              <div className="font-body text-sm text-[var(--text-secondary)] mt-2">True Positive (TP)</div>
            </GlassCard>
            <GlassCard className="flex flex-col items-center justify-center p-6 border border-[var(--state-danger)]/30 bg-[var(--state-danger)]/5">
              <div className="font-mono text-4xl font-bold text-[var(--state-danger)]">
                {data.confusionMatrix.falsePositive}
              </div>
              <div className="font-body text-sm text-[var(--text-secondary)] mt-2">False Positive (FP)</div>
            </GlassCard>
            <GlassCard className="flex flex-col items-center justify-center p-6 border border-[var(--state-danger)]/30 bg-[var(--state-danger)]/5">
              <div className="font-mono text-4xl font-bold text-[var(--state-danger)]">
                {data.confusionMatrix.falseNegative}
              </div>
              <div className="font-body text-sm text-[var(--text-secondary)] mt-2">False Negative (FN)</div>
            </GlassCard>
            <GlassCard className="flex flex-col items-center justify-center p-6 border border-[var(--state-success)]/30 bg-[var(--state-success)]/5">
              <div className="font-mono text-4xl font-bold text-[var(--state-success)]">
                {data.confusionMatrix.trueNegative}
              </div>
              <div className="font-body text-sm text-[var(--text-secondary)] mt-2">True Negative (TN)</div>
            </GlassCard>
          </div>
        </GlassCard>
      </motion.div>
    );
  };

  return (
    <motion.div variants={pageTransition} initial="initial" animate="animate" exit="exit" className="p-6 max-w-6xl mx-auto space-y-6">
      <header className="flex items-center gap-3 mb-8">
        <BarChart3 className="w-8 h-8 text-[var(--accent-primary)]" />
        <h1 className="text-3xl font-display font-bold text-[var(--text-primary)]">Model Evaluation</h1>
      </header>
      
      <div className="flex gap-2 mb-6 bg-[var(--bg-2)] p-1 rounded-lg w-fit border border-[var(--bg-3)]">
        {(['random_forest', 'xgboost', 'lightgbm'] as const).map((model) => (
          <button
            key={model}
            onClick={() => setModelName(model)}
            className={`px-4 py-2 rounded-md font-body text-sm transition-all duration-300 ${
              modelName === model 
                ? 'bg-[var(--accent-primary)]/20 text-[var(--accent-primary)] border border-[var(--accent-primary)]/50 shadow-[0_0_15px_rgba(0,229,255,0.2)]' 
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-3)]'
            }`}
          >
            {model === 'random_forest' ? 'Random Forest' : model === 'xgboost' ? 'XGBoost' : 'LightGBM'}
          </button>
        ))}
      </div>

      {renderContent()}
    </motion.div>
  );
};
