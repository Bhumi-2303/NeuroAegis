import { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { Cpu } from 'lucide-react';

interface Props {
  modelName?: string;
  datasetName?: string;
}

export function ModelInfoCard({ modelName, datasetName }: Props) {
  const [metricsData, setMetricsData] = useState<any>(null);
  
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const res = await fetch(`${API_URL}/api/v1/metrics`);
        const data = await res.json();
        setMetricsData(data);
      } catch (err) {
        console.error('Failed to fetch metrics', err);
      }
    };
    fetchMetrics();
  }, []);

  const defaultStats = metricsData?.average_metrics?.default_threshold;
  
  let avgThreshold = 0.5;
  if (metricsData?.folds && metricsData.folds.length > 0) {
    const thresholds = metricsData.folds.map((f: any) => f.tuned_threshold.threshold);
    avgThreshold = thresholds.reduce((a: number, b: number) => a + b, 0) / thresholds.length;
  }

  const metrics = [
    { label: 'Accuracy', value: defaultStats ? `${(defaultStats.accuracy * 100).toFixed(1)}%` : '--%' },
    { label: 'Precision', value: defaultStats ? `${(defaultStats.precision * 100).toFixed(1)}%` : '--%' },
    { label: 'Recall', value: defaultStats ? `${(defaultStats.recall * 100).toFixed(1)}%` : '--%' },
    { label: 'F1 Score', value: defaultStats ? `${(defaultStats.f1 * 100).toFixed(1)}%` : '--%' },
    { label: 'ROC-AUC', value: defaultStats ? defaultStats.roc_auc.toFixed(3) : '--' },
    { label: 'Features', value: '14' },
  ];

  return (
    <WidgetCard title="Model Information" icon={<Cpu size={16} />}>
      <div className="mb-6">
        <div className="text-xl font-bold text-[var(--text-primary)] capitalize tracking-tight">
          {modelName?.replace(/_/g, ' ') || 'CHB-MIT LightGBM'}
        </div>
        <div className="flex gap-2 items-center mt-2">
          <span className="px-2 py-0.5 bg-[var(--accent-primary)]/10 text-[var(--accent-primary)] border border-[var(--accent-primary)]/30 rounded text-[11px] font-semibold">v2.1.0</span>
          <span className="text-xs text-[var(--text-secondary)] font-medium">Core Prediction Engine</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2.5">
        {metrics.map(m => (
          <div key={m.label} className="p-3 rounded-xl border border-[var(--bg-4)] bg-[var(--bg-3)] flex flex-col justify-center transition-colors hover:bg-[var(--bg-4)]">
            <div className="text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider mb-1">{m.label}</div>
            <div className="text-sm font-semibold text-[var(--text-primary)] font-mono">{m.value}</div>
          </div>
        ))}
      </div>
      
      <div className="mt-5 grid grid-cols-2 gap-4 text-xs font-medium text-[var(--text-secondary)] border-t border-[var(--bg-4)] pt-4">
         <div>
           <span className="text-[var(--text-muted)] block mb-1 text-[10px] uppercase tracking-wider">Threshold</span> 
           {datasetName?.toLowerCase().includes('chb') 
             ? `${avgThreshold.toFixed(2)} (Patient-Tuned)` 
             : '0.50 (Default)'}
         </div>
         <div>
           <span className="text-[var(--text-muted)] block mb-1 text-[10px] uppercase tracking-wider">Inference SLA</span> 
           &lt; 50ms
         </div>
      </div>
    </WidgetCard>
  );
}
