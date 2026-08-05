import { WidgetCard } from './WidgetCard';
import { ConfidenceGauge } from '../../../../shared/components';
import { ShieldAlert, ShieldCheck, Clock, Database, Cpu } from 'lucide-react';

interface Props {
  probability: number;
  label: string;
  datasetName?: string;
  modelName?: string;
  inferenceTime?: string;
}

export function PredictionSummaryCard({ probability, label, datasetName, modelName, inferenceTime = '42ms' }: Props) {
  const isSeizure = label === 'seizure';
  const confValue = Math.max(probability, 1 - probability) * 100;

  return (
    <WidgetCard>
      <div className="flex flex-col lg:flex-row items-center gap-8 h-full justify-center py-4">
        
        {/* Left: Gauge */}
        <div className="flex-shrink-0 flex flex-col items-center">
          <ConfidenceGauge value={confValue} size={180} isSeizure={isSeizure} />
          <div className={`mt-6 flex items-center gap-2 px-4 py-2 rounded-full border ${isSeizure ? 'bg-[var(--state-danger)]/10 border-[var(--state-danger)]/30 text-[var(--state-danger)]' : 'bg-[var(--state-success)]/10 border-[var(--state-success)]/30 text-[var(--state-success)]'}`}>
            {isSeizure ? <ShieldAlert size={18} /> : <ShieldCheck size={18} />}
            <span className="text-sm font-bold tracking-wide uppercase">
              {isSeizure ? 'Seizure Detected' : 'Normal Activity'}
            </span>
          </div>
        </div>

        {/* Right: Details */}
        <div className="flex-1 w-full space-y-6">
          <div>
            <div className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">Risk Assessment</div>
            <div className="flex items-center justify-between text-sm mb-1.5">
              <span className="font-medium text-[var(--text-secondary)]">Seizure Probability</span>
              <span className="font-mono font-bold text-[var(--text-primary)]">{(probability * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 w-full bg-[var(--bg-4)] rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full transition-all duration-1000 ${isSeizure ? 'bg-[var(--state-danger)]' : 'bg-[var(--state-success)]'}`} 
                style={{ width: `${probability * 100}%` }}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
             <div className="bg-[var(--bg-3)] p-3 rounded-xl border border-[var(--bg-4)] flex items-center gap-3">
               <div className="p-2 bg-[var(--bg-2)] rounded-lg shadow-sm border border-[var(--bg-4)] text-[var(--text-muted)]">
                 <Database size={16} />
               </div>
               <div>
                 <div className="text-[10px] uppercase font-bold text-[var(--text-muted)]">Dataset</div>
                 <div className="text-xs font-semibold text-[var(--text-primary)] capitalize">{datasetName || 'Unknown'}</div>
               </div>
             </div>
             
             <div className="bg-[var(--bg-3)] p-3 rounded-xl border border-[var(--bg-4)] flex items-center gap-3">
               <div className="p-2 bg-[var(--bg-2)] rounded-lg shadow-sm border border-[var(--bg-4)] text-[var(--text-muted)]">
                 <Cpu size={16} />
               </div>
               <div>
                 <div className="text-[10px] uppercase font-bold text-[var(--text-muted)]">Model</div>
                 <div className="text-xs font-semibold text-[var(--text-primary)] capitalize">{modelName?.replace('_', ' ') || 'XGBoost'}</div>
               </div>
             </div>
             
             <div className="bg-[var(--bg-3)] p-3 rounded-xl border border-[var(--bg-4)] flex items-center gap-3 col-span-2">
               <div className="p-2 bg-[var(--bg-2)] rounded-lg shadow-sm border border-[var(--bg-4)] text-[var(--text-muted)]">
                 <Clock size={16} />
               </div>
               <div>
                 <div className="text-[10px] uppercase font-bold text-[var(--text-muted)]">Inference Time</div>
                 <div className="text-xs font-semibold text-[var(--text-primary)] font-mono">{inferenceTime}</div>
               </div>
             </div>
          </div>
        </div>

      </div>
    </WidgetCard>
  );
}
