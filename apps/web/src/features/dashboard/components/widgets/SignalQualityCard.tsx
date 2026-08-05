import { WidgetCard } from './WidgetCard';
import { Activity } from 'lucide-react';
import { motion } from 'framer-motion';

export function SignalQualityCard() {
  const quality = 94; // Mock
  const noise = 6;
  const drift = 2;
  const missing = 0;
  
  const isExcellent = quality > 90;

  return (
    <WidgetCard title="Signal Quality Assessment" icon={<Activity size={16} />}>
      <div className="flex justify-between items-start mb-6">
        <div>
          <div className="text-3xl font-bold text-[var(--text-primary)] tracking-tight">{quality}%</div>
          <div className="text-sm text-[var(--text-secondary)] mt-0.5">Overall Quality Index</div>
        </div>
        <div className={`px-2.5 py-1 rounded-full text-[11px] font-semibold border shadow-sm ${isExcellent ? 'bg-[var(--state-success)]/10 border-[var(--state-success)]/30 text-[var(--state-success)]' : 'bg-[var(--state-warning)]/10 border-[var(--state-warning)]/30 text-[var(--state-warning)]'}`}>
          {isExcellent ? 'Excellent Quality' : 'Poor Quality'}
        </div>
      </div>

      <div className="space-y-5 flex-1 justify-center flex flex-col mt-2">
        <QualityBar label="Noise Percentage" value={noise} invertColor />
        <QualityBar label="Baseline Drift" value={drift} invertColor />
        <QualityBar label="Missing Samples" value={missing} invertColor />
      </div>
    </WidgetCard>
  );
}

function QualityBar({ label, value, invertColor = false }: { label: string, value: number, invertColor?: boolean }) {
  const isBad = invertColor ? value > 10 : value < 90;
  const color = isBad ? 'bg-[var(--state-warning)]' : 'bg-[var(--state-success)]';

  return (
    <div>
      <div className="flex justify-between text-xs font-medium text-[var(--text-secondary)] mb-2">
        <span>{label}</span>
        <span className="font-mono text-[var(--text-primary)]">{value}%</span>
      </div>
      <div className="h-2 w-full bg-[var(--bg-4)] rounded-full overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className={`h-full ${color} rounded-full`}
        />
      </div>
    </div>
  );
}
