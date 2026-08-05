import { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, FileText, Download, Calendar, User, Activity, CheckCircle, Search, Target } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { GlassCard } from '../../../shared/components';
import { pageTransition, fadeIn } from '../../../shared/lib/motion-presets';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const ReportsPage = () => {
  const [activeTab, setActiveTab] = useState<'generate' | 'metrics'>('generate');
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportSuccess, setReportSuccess] = useState(false);

  const { data: metricsData, isLoading, isError } = useQuery({
    queryKey: ['model-metrics'],
    queryFn: async () => {
      const res = await fetch(`${API_URL}/api/v1/metrics`);
      if (!res.ok) throw new Error('Failed to fetch metrics');
      return res.json();
    },
    enabled: activeTab === 'metrics'
  });

  const handleGenerateReport = (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);
    setReportSuccess(false);
    setTimeout(() => {
      setIsGenerating(false);
      setReportSuccess(true);
      setTimeout(() => setReportSuccess(false), 5000);
    }, 2000);
  };

  const renderMetrics = () => {
    if (isLoading) {
      return (
        <div className="mt-6 flex flex-col items-center justify-center h-[400px] text-center">
          <div className="w-8 h-8 border-4 border-[var(--accent-primary)] border-t-transparent rounded-full animate-spin mb-4" />
          <p className="text-sm text-gray-500">Loading evaluation metrics...</p>
        </div>
      );
    }
    
    if (isError || !metricsData) {
      return (
        <div className="mt-6 flex flex-col items-center justify-center h-[400px] text-center text-red-500">
          <AlertCircle size={32} className="mb-4" />
          <p>Failed to load validation metrics from backend.</p>
        </div>
      );
    }

    const { metadata, average_metrics, folds } = metricsData;

    return (
      <motion.div variants={fadeIn} initial="initial" animate="animate" className="mt-6 space-y-6">
        <GlassCard title="Clinical Validation Overview" className="p-6">
          <div className="flex items-center gap-4 mb-6 pb-4 border-b border-gray-100">
            <div>
              <h3 className="font-semibold text-gray-800">{metadata?.validation_strategy}</h3>
              <p className="text-sm text-gray-500">{metadata?.description} — Sample Size: <span className="font-mono font-medium text-indigo-600">{metadata?.sample_size}</span></p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-2"><Target size={16}/> Default Threshold (0.5)</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-xs text-gray-500 mb-1">ROC AUC</div>
                  <div className="text-xl font-bold text-gray-900">{(average_metrics.default_threshold.roc_auc * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-xs text-gray-500 mb-1">Accuracy</div>
                  <div className="text-xl font-bold text-gray-900">{(average_metrics.default_threshold.accuracy * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-xs text-gray-500 mb-1">Precision</div>
                  <div className="text-xl font-bold text-gray-900">{(average_metrics.default_threshold.precision * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-xs text-gray-500 mb-1">Recall (Sensitivity)</div>
                  <div className="text-xl font-bold text-gray-900">{(average_metrics.default_threshold.recall * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-indigo-700 flex items-center gap-2"><Activity size={16}/> Threshold-Tuned (Optimal F1)</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-indigo-50/50 rounded-lg p-4 border border-indigo-100">
                  <div className="text-xs text-indigo-600/70 mb-1">F1 Score</div>
                  <div className="text-xl font-bold text-indigo-900">{(average_metrics.tuned_threshold.f1 * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-indigo-50/50 rounded-lg p-4 border border-indigo-100">
                  <div className="text-xs text-indigo-600/70 mb-1">Precision</div>
                  <div className="text-xl font-bold text-indigo-900">{(average_metrics.tuned_threshold.precision * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-indigo-50/50 rounded-lg p-4 border border-indigo-100 col-span-2">
                  <div className="text-xs text-indigo-600/70 mb-1">Recall (Sensitivity)</div>
                  <div className="text-xl font-bold text-indigo-900">{(average_metrics.tuned_threshold.recall * 100).toFixed(1)}%</div>
                  <div className="text-xs text-indigo-500 mt-1">Tuned for max seizure detection</div>
                </div>
              </div>
            </div>
          </div>
        </GlassCard>

        <GlassCard title="Per-Fold Breakdown (Leave-One-Patient-Out)" className="p-6">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-gray-500 uppercase bg-gray-50">
                <tr>
                  <th className="px-4 py-3 rounded-tl-lg">Patient ID</th>
                  <th className="px-4 py-3">Test Seizures</th>
                  <th className="px-4 py-3 text-center border-l border-gray-200" colSpan={3}>Default (AUC / Prec / Rec)</th>
                  <th className="px-4 py-3 text-center border-l border-gray-200 rounded-tr-lg" colSpan={3}>Tuned (Thresh / Prec / Rec)</th>
                </tr>
              </thead>
              <tbody>
                {folds.map((fold: any, i: number) => (
                  <tr key={i} className="border-b border-gray-100 last:border-0 hover:bg-gray-50/50">
                    <td className="px-4 py-3 font-medium text-gray-900">{fold.patient_id}</td>
                    <td className="px-4 py-3 text-gray-600">{fold.test_seizures}</td>
                    <td className="px-4 py-3 border-l border-gray-200">{(fold.default_threshold.roc_auc * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3 text-gray-500">{(fold.default_threshold.precision * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3 text-gray-500">{(fold.default_threshold.recall * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3 border-l border-gray-200 text-indigo-600">{fold.tuned_threshold.threshold.toFixed(2)}</td>
                    <td className="px-4 py-3 text-indigo-600/70">{(fold.tuned_threshold.precision * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3 text-indigo-600/70">{(fold.tuned_threshold.recall * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
                  <CheckCircle className="w-4 h-4" /> Report generated successfully!
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
