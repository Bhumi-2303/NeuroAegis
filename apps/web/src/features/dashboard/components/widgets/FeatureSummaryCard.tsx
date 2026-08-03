import { WidgetCard } from './WidgetCard';
import { Network, Activity, Waves, Zap, Hash, AlignCenter, Calculator } from 'lucide-react';

export function FeatureSummaryCard() {
  const categories = [
    { name: 'Temporal Features', count: 12, icon: <Activity size={14} />, desc: 'Time-domain statistical moments' },
    { name: 'Frequency Features', count: 18, icon: <Waves size={14} />, desc: 'Spectral power distributions' },
    { name: 'Wavelet Features', count: 24, icon: <Network size={14} />, desc: 'Multi-resolution analysis' },
    { name: 'Entropy Features', count: 4, icon: <Hash size={14} />, desc: 'Signal complexity and chaos' },
    { name: 'Hjorth Parameters', count: 3, icon: <AlignCenter size={14} />, desc: 'Activity, mobility, complexity' },
    { name: 'Band Power', count: 5, icon: <Zap size={14} />, desc: 'Alpha, Beta, Theta, Delta, Gamma' },
  ];

  const total = categories.reduce((acc, curr) => acc + curr.count, 0);

  return (
    <WidgetCard title="Feature Engineering Summary" icon={<Calculator size={16} />}>
      <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-100">
        <div>
          <div className="text-2xl font-bold text-gray-900">{total}</div>
          <div className="text-xs text-gray-500 font-medium">Total Extracted Features</div>
        </div>
        <div className="h-10 w-10 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center">
          <Network size={20} />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {categories.map(cat => (
          <div key={cat.name} className="flex items-start gap-3 p-3 rounded-xl border border-gray-100 bg-white hover:border-gray-200 transition-colors shadow-sm">
            <div className="p-2 bg-gray-50 text-gray-500 rounded-lg shrink-0">
              {cat.icon}
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex justify-between items-center mb-0.5">
                <span className="text-xs font-bold text-gray-700 truncate">{cat.name}</span>
                <span className="text-[10px] font-mono font-medium text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">{cat.count}</span>
              </div>
              <p className="text-[10px] text-gray-400 truncate">{cat.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </WidgetCard>
  );
}
