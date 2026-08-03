import { WidgetCard } from './WidgetCard';
import { Cpu } from 'lucide-react';

interface Props {
  modelName?: string;
}

export function ModelInfoCard({ modelName }: Props) {
  const metrics = [
    { label: 'Accuracy', value: '98.2%' },
    { label: 'Precision', value: '97.9%' },
    { label: 'Recall', value: '98.5%' },
    { label: 'F1 Score', value: '98.2%' },
    { label: 'ROC-AUC', value: '0.994' },
    { label: 'Features', value: '14' },
  ];

  return (
    <WidgetCard title="Model Information" icon={<Cpu size={16} />}>
      <div className="mb-6">
        <div className="text-xl font-bold text-gray-900 capitalize tracking-tight">
          {modelName?.replace('_', ' ') || 'CHB-MIT XGBoost'}
        </div>
        <div className="flex gap-2 items-center mt-2">
          <span className="px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-100 rounded text-[11px] font-semibold">v2.1.0</span>
          <span className="text-xs text-gray-500 font-medium">Core Prediction Engine</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2.5">
        {metrics.map(m => (
          <div key={m.label} className="p-3 rounded-xl border border-gray-100 bg-gray-50/50 flex flex-col justify-center transition-colors hover:bg-gray-50">
            <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-1">{m.label}</div>
            <div className="text-sm font-semibold text-gray-900 font-mono">{m.value}</div>
          </div>
        ))}
      </div>
      
      <div className="mt-5 grid grid-cols-2 gap-4 text-xs font-medium text-gray-600 border-t border-gray-100 pt-4">
         <div><span className="text-gray-400 block mb-1 text-[10px] uppercase tracking-wider">Threshold</span> 0.50 (Optimized)</div>
         <div><span className="text-gray-400 block mb-1 text-[10px] uppercase tracking-wider">Inference SLA</span> &lt; 50ms</div>
      </div>
    </WidgetCard>
  );
}
