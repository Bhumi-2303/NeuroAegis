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
          <ConfidenceGauge value={confValue} size={180} />
          <div className={`mt-6 flex items-center gap-2 px-4 py-2 rounded-full border ${isSeizure ? 'bg-red-50 border-red-200 text-red-700' : 'bg-emerald-50 border-emerald-200 text-emerald-700'}`}>
            {isSeizure ? <ShieldAlert size={18} /> : <ShieldCheck size={18} />}
            <span className="text-sm font-bold tracking-wide uppercase">
              {isSeizure ? 'Seizure Detected' : 'Normal Activity'}
            </span>
          </div>
        </div>

        {/* Right: Details */}
        <div className="flex-1 w-full space-y-6">
          <div>
            <div className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Risk Assessment</div>
            <div className="flex items-center justify-between text-sm mb-1.5">
              <span className="font-medium text-gray-700">Seizure Probability</span>
              <span className="font-mono font-bold text-gray-900">{(probability * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full transition-all duration-1000 ${isSeizure ? 'bg-red-500' : 'bg-emerald-500'}`} 
                style={{ width: `${probability * 100}%` }}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
             <div className="bg-gray-50 p-3 rounded-xl border border-gray-100 flex items-center gap-3">
               <div className="p-2 bg-white rounded-lg shadow-sm border border-gray-50 text-gray-400">
                 <Database size={16} />
               </div>
               <div>
                 <div className="text-[10px] uppercase font-bold text-gray-400">Dataset</div>
                 <div className="text-xs font-semibold text-gray-900 capitalize">{datasetName || 'Unknown'}</div>
               </div>
             </div>
             
             <div className="bg-gray-50 p-3 rounded-xl border border-gray-100 flex items-center gap-3">
               <div className="p-2 bg-white rounded-lg shadow-sm border border-gray-50 text-gray-400">
                 <Cpu size={16} />
               </div>
               <div>
                 <div className="text-[10px] uppercase font-bold text-gray-400">Model</div>
                 <div className="text-xs font-semibold text-gray-900 capitalize">{modelName?.replace('_', ' ') || 'XGBoost'}</div>
               </div>
             </div>
             
             <div className="bg-gray-50 p-3 rounded-xl border border-gray-100 flex items-center gap-3 col-span-2">
               <div className="p-2 bg-white rounded-lg shadow-sm border border-gray-50 text-gray-400">
                 <Clock size={16} />
               </div>
               <div>
                 <div className="text-[10px] uppercase font-bold text-gray-400">Inference Time</div>
                 <div className="text-xs font-semibold text-gray-900 font-mono">{inferenceTime}</div>
               </div>
             </div>
          </div>
        </div>

      </div>
    </WidgetCard>
  );
}
