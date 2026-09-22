import React from 'react';
import { ShapExplanation } from '../../types/neuroaegis';
import { HelpCircle } from 'lucide-react';

interface ShapExplanationViewProps {
  readonly explanation: ShapExplanation | null;
}

export const ShapExplanationView: React.FC<ShapExplanationViewProps> = ({ explanation }) => {
  if (!explanation) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 lg:p-5 shadow-lg">
        <h3 className="text-xs uppercase font-bold tracking-wider text-slate-400">XAI biomarkers (SHAP)</h3>
        <p className="text-xs text-slate-500 mt-3">No SHAP explanation available.</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 lg:p-5 space-y-4 shadow-lg">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-1.5">
          <h3 className="text-xs uppercase font-bold tracking-wider text-slate-400">
            XAI Biomarkers (SHAP)
          </h3>
          <div className="group relative cursor-pointer" aria-label="Explainable AI Details">
            <HelpCircle className="h-3.5 w-3.5 text-slate-500 hover:text-slate-300" />
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-48 bg-slate-950 text-slate-300 text-[10px] p-2 rounded shadow-xl border border-slate-800 z-50">
              Positive values push the model towards predicting seizure; negative values push towards baseline.
            </div>
          </div>
        </div>
        <span className="text-[10px] font-mono text-slate-400">
          Base: {explanation.baseValue.toFixed(2)}
        </span>
      </div>

      <p className="text-[10px] text-slate-500">Feature-level explanation only. SHAP values are not channel-level seizure localization.</p>

      <div className="space-y-3 pt-1">
        {explanation.features.map((feat) => {
          const isPositive = feat.contribution > 0;
          const absContribution = Math.min(100, Math.abs(feat.contribution) * 180);

          return (
            <div key={`feat-${feat.featureName}`} className="space-y-1 text-xs">
              <div className="flex justify-between text-slate-300">
                <span className="font-medium text-slate-300">{feat.featureName}</span>
                <span
                  className={`font-mono text-[11px] font-bold ${
                    isPositive ? 'text-rose-400' : 'text-emerald-400'
                  }`}
                >
                  {isPositive ? '+' : ''}
                  {feat.contribution.toFixed(2)}
                </span>
              </div>
              {feat.rawValue !== undefined && (
                <div className="text-[10px] text-slate-500 font-mono">
                  Raw value: {feat.rawValue.toFixed(3)}
                  {feat.referenceRange && ` | Reference: ${feat.referenceRange[0].toFixed(3)}-${feat.referenceRange[1].toFixed(3)}`}
                </div>
              )}

              {/* Bi-directional contribution bar */}
              <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden flex">
                <div className="w-1/2 flex justify-end">
                  {!isPositive && (
                    <div
                      className="h-full bg-emerald-500 rounded-l transition-all duration-300"
                      style={{ width: `${absContribution}%` }}
                    />
                  )}
                </div>
                <div className="w-1/2 flex justify-start">
                  {isPositive && (
                    <div
                      className="h-full bg-rose-500 rounded-r transition-all duration-300"
                      style={{ width: `${absContribution}%` }}
                    />
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
