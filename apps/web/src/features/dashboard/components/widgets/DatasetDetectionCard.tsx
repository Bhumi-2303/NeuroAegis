import { WidgetCard } from './WidgetCard';
import { Database, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';

interface Props {
  datasetName?: string;
  confidence?: number;
  samplingRate?: string;
  channels?: string;
}

export function DatasetDetectionCard({ datasetName, confidence = 0.99, samplingRate, channels }: Props) {
  const isBonn = datasetName?.toLowerCase().includes('bonn');
  const reasons = isBonn ? [
    'Single Channel Structure',
    '173.61 Hz Expected Frequency',
    'Signal Length 4097 samples'
  ] : [
    'Multi-channel Structure',
    '256 Hz Expected Frequency',
    'Typical CHB-MIT Duration Pattern'
  ];

  return (
    <WidgetCard title="Dataset Detection" icon={<Database size={16} />}>
      <div className="flex flex-col h-full gap-4">
        
        <div className="flex items-start justify-between">
          <div>
            <div className="text-2xl font-bold text-gray-900 capitalize tracking-tight">
              {datasetName || 'Unknown'} EEG
            </div>
            <div className="text-sm text-gray-500 mt-0.5">Automatic signature match</div>
          </div>
          
          <div className="flex flex-col items-end">
            <div className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 shadow-sm">
              Confidence {(confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 mt-1">
          <div className="p-3 bg-gray-50/80 rounded-xl border border-gray-100">
            <div className="text-xs text-gray-500 mb-1 font-medium">Detected Rate</div>
            <div className="font-mono text-sm text-gray-900">{samplingRate || 'N/A'}</div>
          </div>
          <div className="p-3 bg-gray-50/80 rounded-xl border border-gray-100">
            <div className="text-xs text-gray-500 mb-1 font-medium">Channels</div>
            <div className="font-mono text-sm text-gray-900">{channels ? channels.split(',').length : 'N/A'}</div>
          </div>
        </div>

        <div className="mt-2 pt-4 border-t border-gray-100">
          <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-3">Matching Criteria</div>
          <div className="space-y-2.5">
            {reasons.map((reason, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 * i }}
                className="flex items-center gap-2.5 text-sm text-gray-700"
              >
                <CheckCircle2 size={16} className="text-emerald-500 shrink-0" />
                <span className="font-medium text-gray-600">{reason}</span>
              </motion.div>
            ))}
          </div>
        </div>

      </div>
    </WidgetCard>
  );
}
