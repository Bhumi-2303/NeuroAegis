import { WidgetCard } from './WidgetCard';
import { Activity, Lock } from 'lucide-react';

interface Props {
  data: number[];
}

export function RawFilteredComparison({ data }: Props) {
  return (
    <WidgetCard title="Preprocessing Comparison" icon={<Activity size={16} />}>
      <div className="flex flex-col items-center justify-center h-[250px] text-center border-2 border-dashed border-[var(--bg-4)] rounded-xl bg-[var(--bg-3)]">
        <Lock className="text-[var(--text-muted)] mb-3" size={28} />
        <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-1">Awaiting Backend Integration</h3>
        <p className="text-xs text-[var(--text-secondary)] max-w-[200px]">
          Preprocessing comparisons will be available when the backend API provides filtered signal data.
        </p>
      </div>
    </WidgetCard>
  );
}
