import { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, FileText, Download, Calendar, User } from 'lucide-react';
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
  const [activeTab, setActiveTab] = useState<'generate' | 'metrics'>('generate');
  const { data, isLoading, isError, error, refetch } = useEvaluationMetrics(modelName);

  const [isGenerating, setIsGenerating] = useState(false);
  const [reportSuccess, setReportSuccess] = useState(false);

  const handleGenerateReport = (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);
    setReportSuccess(false);
    
    // Simulate API call
    setTimeout(() => {
      setIsGenerating(false);
      setReportSuccess(true);
      
      setTimeout(() => setReportSuccess(false), 5000);
    }, 2000);
  };

  const renderMetrics = () => {
    if (isLoading) {
      return (
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mt-6">
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
        <div className="mt-6">
          <ErrorState
            title="Failed to load metrics"
            message={error instanceof Error ? error.message : 'An unknown error occurred'}
            onRetry={refetch}
          />
        </div>
      );
    }

    if (!data) {
      return (
        <div className="mt-6">
          <EmptyState
            icon={AlertCircle}
            title="No Data Available"
            description="Evaluation metrics for the selected model are not available."
          />
        </div>
      );
    }

    const formatPercent = (val: number) => `${(val * 100).toFixed(1)}%`;

    return (
      <motion.div variants={fadeIn} initial="initial" animate="animate" className="space-y-6 mt-6">
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
      </motion.div>
    );
  };

  const renderGenerateForm = () => (
    <motion.div variants={fadeIn} initial="initial" animate="animate" className="mt-6 max-w-2xl mx-auto">
      <GlassCard title="Generate Clinical Report" className="p-8">
        <form onSubmit={handleGenerateReport} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-[var(--font-body)] text-[var(--text-secondary)] flex items-center gap-2">
                <User className="w-4 h-4" /> Patient ID
              </label>
              <input 
                type="text" 
                required
                placeholder="e.g. PAT-12345"
                className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-3 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)] transition-colors"
              />
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-[var(--font-body)] text-[var(--text-secondary)] flex items-center gap-2">
                <FileText className="w-4 h-4" /> Report Type
              </label>
              <select className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-3 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)] transition-colors appearance-none">
                <option value="summary">Summary Report</option>
                <option value="detailed">Detailed Analysis</option>
                <option value="clinical">Clinical Diagnostics</option>
              </select>
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-[var(--font-body)] text-[var(--text-secondary)] flex items-center gap-2">
                <Calendar className="w-4 h-4" /> Start Date
              </label>
              <input 
                type="date" 
                className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-3 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)] transition-colors"
              />
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-[var(--font-body)] text-[var(--text-secondary)] flex items-center gap-2">
                <Calendar className="w-4 h-4" /> End Date
              </label>
              <input 
                type="date" 
                className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-3 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)] transition-colors"
              />
            </div>
          </div>
          
          <div className="space-y-2 pt-2">
            <label className="text-sm font-[var(--font-body)] text-[var(--text-secondary)]">Additional Notes</label>
            <textarea 
              rows={4}
              placeholder="Any specific findings to include in the report..."
              className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-3 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)] transition-colors custom-scrollbar"
            />
          </div>

          <div className="pt-4 border-t border-[var(--bg-3)] flex items-center justify-between">
            <div>
              {reportSuccess && (
                <motion.span 
                  initial={{ opacity: 0, y: 10 }} 
                  animate={{ opacity: 1, y: 0 }} 
                  className="text-sm text-[var(--state-success)] font-[var(--font-body)] flex items-center gap-2"
                >
                  <AlertCircle className="w-4 h-4" /> Report generated successfully!
                </motion.span>
              )}
            </div>
            <button 
              type="submit"
              disabled={isGenerating}
              className="px-6 py-3 bg-[var(--accent-primary)] text-[var(--bg-1)] rounded-lg font-[var(--font-body)] text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center gap-2"
            >
              {isGenerating ? (
                <>
                  <div className="w-4 h-4 border-2 border-[var(--bg-1)] border-t-transparent rounded-full animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  Generate Report
                </>
              )}
            </button>
          </div>
        </form>
      </GlassCard>
    </motion.div>
  );

  return (
    <motion.div variants={pageTransition} initial="initial" animate="animate" exit="exit" className="p-6 max-w-6xl mx-auto space-y-6">
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <FileText className="w-8 h-8 text-[var(--accent-primary)]" />
          <h1 className="text-3xl font-display font-bold text-[var(--text-primary)]">Reports & Analytics</h1>
        </div>
        
        <div className="flex gap-2 bg-[var(--bg-2)] p-1 rounded-lg border border-[var(--bg-3)]">
          <button
            onClick={() => setActiveTab('generate')}
            className={`px-4 py-2 rounded-md font-body text-sm transition-all duration-300 ${
              activeTab === 'generate' 
                ? 'bg-[var(--bg-3)] text-[var(--text-primary)] shadow-sm' 
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            Generate Report
          </button>
          <button
            onClick={() => setActiveTab('metrics')}
            className={`px-4 py-2 rounded-md font-body text-sm transition-all duration-300 ${
              activeTab === 'metrics' 
                ? 'bg-[var(--bg-3)] text-[var(--text-primary)] shadow-sm' 
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            Model Metrics
          </button>
        </div>
      </header>

      {activeTab === 'generate' ? renderGenerateForm() : renderMetrics()}
    </motion.div>
  );
};
