import { WidgetCard } from './WidgetCard';
import { ActivitySquare, Lock } from 'lucide-react';

export function TimeFrequencyAnalysis() {
  return (
    <WidgetCard title="Time-Frequency Analysis" icon={<ActivitySquare size={16} />}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PlaceholderPanel title="Spectrogram Analysis" />
        <PlaceholderPanel title="Wavelet Scalogram" />
        <PlaceholderPanel title="Power Spectral Density" />
        <PlaceholderPanel title="Relative Band Power" />
      </div>
    </WidgetCard>
  );
}

function PlaceholderPanel({ title }: { title: string }) {
  return (
    <div className="h-32 border border-dashed border-gray-200 rounded-xl bg-gray-50 flex flex-col items-center justify-center relative overflow-hidden group">
      <div className="absolute inset-0 bg-gradient-to-br from-gray-50/50 to-white/50 backdrop-blur-[1px] z-10" />
      <div className="relative z-20 flex flex-col items-center">
        <Lock size={16} className="text-gray-300 mb-2" />
        <span className="text-xs font-semibold text-gray-500">{title}</span>
        <span className="text-[10px] text-gray-400 mt-1">Awaiting Backend Integration</span>
      </div>
    </div>
  );
}
