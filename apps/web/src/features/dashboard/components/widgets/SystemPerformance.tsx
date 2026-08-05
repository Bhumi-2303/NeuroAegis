import { WidgetCard } from './WidgetCard';
import { Server } from 'lucide-react';

export function SystemPerformance() {
  const metrics = [
    { label: 'CPU Usage', value: '12.4%', sub: '4 Cores Active' },
    { label: 'Memory', value: '142 MB', sub: 'Heap Allocation' },
    { label: 'GPU Status', value: 'Idle', sub: 'Fallback to CPU' },
    { label: 'Model Size', value: '2.4 MB', sub: 'In-memory' },
    { label: 'Feature Extraction', value: '18ms', sub: 'Pandas/NumPy' },
    { label: 'Prediction', value: '8ms', sub: 'XGBoost Eval' },
  ];

  return (
    <WidgetCard title="System Performance" icon={<Server size={16} />}>
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
        {metrics.map(m => (
          <div key={m.label} className="p-3 bg-[var(--bg-3)] rounded-xl border border-[var(--bg-4)] flex flex-col justify-center">
            <span className="text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider mb-1">{m.label}</span>
            <span className="text-sm font-semibold text-[var(--text-primary)] font-mono">{m.value}</span>
            <span className="text-[10px] text-[var(--text-muted)] mt-1">{m.sub}</span>
          </div>
        ))}
      </div>
    </WidgetCard>
  );
}
