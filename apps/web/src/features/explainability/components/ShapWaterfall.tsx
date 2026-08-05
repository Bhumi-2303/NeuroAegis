import { motion } from 'framer-motion';
import { GlassCard } from '../../../shared/components';

export interface ShapFeature {
  featureName: string;
  value: number;
  rawValue?: number;
  referenceRange?: [number, number];
}

export interface ShapWaterfallProps {
  baseValue: number;
  features: ShapFeature[];
  finalProbability: number;
}

export function ShapWaterfall({ baseValue, features, finalProbability }: ShapWaterfallProps) {
  // We sort by absolute SHAP value to show largest impacts first
  const sortedFeatures = [...features].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  
  // Calculate running totals for the waterfall steps
  let currentValue = baseValue;
  const steps = sortedFeatures.map((f) => {
    const start = currentValue;
    currentValue += f.value;
    return { ...f, start, end: currentValue };
  });

  // Calculate the scale for the X axis
  const allValues = [baseValue, finalProbability, ...steps.map(s => s.start), ...steps.map(s => s.end)];
  const minVal = Math.min(...allValues) - 0.5;
  const maxVal = Math.max(...allValues) + 0.5;
  const range = maxVal - minVal;

  const getPercentage = (val: number) => Math.max(0, Math.min(100, ((val - minVal) / range) * 100));

  return (
    <GlassCard title="SHAP Feature Explanations" className="p-6">
      <div className="flex flex-col gap-6 w-full text-sm font-[var(--font-body)]">
        
        {/* Legend */}
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1">
             <div className="w-3 h-3 bg-[var(--state-success)]/20 border border-[var(--state-success)] rounded-sm"></div>
             <span className="text-[var(--text-secondary)]">green = pushes toward non-seizure</span>
          </div>
          <div className="flex items-center gap-1">
             <div className="w-3 h-3 bg-[var(--state-danger)]/20 border border-[var(--state-danger)] rounded-sm"></div>
             <span className="text-[var(--text-secondary)]">red = pushes toward seizure</span>
          </div>
        </div>

        {/* Header / Axis info */}
        <div className="flex justify-between text-xs text-[var(--text-secondary)] border-b border-[var(--bg-3)] pb-2">
          <span className="w-1/4">Feature</span>
          <span className="w-1/6">Computed Value</span>
          <span className="w-1/6">Typical Range</span>
          <span className="w-5/12 text-right">Log-Odds Impact</span>
        </div>

        {/* Base Value */}
        <div className="flex items-center gap-4 group">
          <div className="w-1/4 flex items-center justify-between pr-4">
            <span className="font-semibold text-[var(--text-primary)]">Base Rate (Prior)</span>
          </div>
          <div className="w-1/6 text-xs text-[var(--text-secondary)]">
            --
          </div>
          <div className="w-1/6 text-xs text-[var(--text-secondary)]">
            --
          </div>
          <div className="w-5/12 flex-1 relative h-10">
            <div 
              className="absolute h-full border-r-2 border-[var(--text-secondary)] border-dashed"
              style={{ left: `${getPercentage(baseValue)}%` }}
            >
              <span className="absolute -top-5 -translate-x-1/2 text-xs text-[var(--text-secondary)] font-mono">{baseValue.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Feature Contributions */}
        {steps.map((step, idx) => {
          const isPositive = step.value > 0;
          const bgClass = isPositive ? 'bg-[var(--state-danger)]/20' : 'bg-[var(--state-success)]/20';
          const textClass = isPositive ? 'text-[var(--state-danger)]' : 'text-[var(--state-success)]';
          
          const startPct = getPercentage(step.start);
          const endPct = getPercentage(step.end);
          const widthPct = Math.abs(endPct - startPct);
          const leftPct = Math.min(startPct, endPct);

          const hasRaw = step.rawValue !== undefined;
          
          return (
            <div key={step.featureName} className="flex items-center gap-4 group">
              <div className="w-1/4 flex flex-col pr-4">
                <span className="font-medium text-[var(--text-primary)] break-words">{step.featureName}</span>
              </div>
              <div className="w-1/6 flex flex-col text-xs">
                {hasRaw ? (
                   <span className="font-mono text-[var(--text-primary)]">{step.rawValue?.toFixed(4)}</span>
                ) : (
                  <span className="text-[var(--text-secondary)]">--</span>
                )}
              </div>
              <div className="w-1/6 flex flex-col text-xs">
                {step.referenceRange ? (
                   <span className="font-mono text-[var(--text-secondary)] opacity-80">
                     {step.referenceRange[0].toFixed(2)} - {step.referenceRange[1].toFixed(2)}
                   </span>
                ) : (
                  <span className="text-[var(--text-secondary)]">--</span>
                )}
              </div>
              <div className="w-5/12 flex-1 relative h-10">
                <motion.div
                  initial={{ width: 0, x: isPositive ? 0 : '100%' }}
                  animate={{ width: `${widthPct}%`, x: 0 }}
                  transition={{ duration: 0.5, delay: idx * 0.1 }}
                  className={`absolute h-full rounded ${bgClass} border border-current ${textClass} flex items-center ${isPositive ? 'justify-end pr-2' : 'justify-start pl-2'}`}
                  style={{ left: `${leftPct}%` }}
                >
                  <span className="text-[12px] font-mono font-bold whitespace-nowrap">
                    {isPositive ? '+' : ''}{step.value.toFixed(2)}
                  </span>
                </motion.div>
                {/* Connecting line to next step */}
                <div 
                  className="absolute h-full border-r border-[var(--bg-3)] border-dotted top-5"
                  style={{ left: `${endPct}%`, height: '150%' }}
                />
              </div>
            </div>
          );
        })}

        {/* Final Output */}
        <div className="flex items-center gap-4 pt-4 border-t border-[var(--bg-3)]">
          <div className="w-1/4 flex items-center pr-4">
            <span className="font-bold text-[var(--text-primary)]">Probability of Seizure</span>
          </div>
          <div className="w-1/6 text-xs text-[var(--text-secondary)]">
            --
          </div>
          <div className="w-1/6 text-xs text-[var(--text-secondary)]">
            --
          </div>
          <div className="w-5/12 flex-1 relative h-10">
            <div 
              className="absolute h-full border-r-2 border-[var(--accent-primary)]"
              style={{ left: `${getPercentage(currentValue)}%` }}
            >
              <span className="absolute -top-6 -translate-x-1/2 text-sm text-[var(--accent-primary)] font-mono font-bold">
                {(finalProbability * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

      </div>
    </GlassCard>
  );
}
