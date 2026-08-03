import React from 'react';
import { WidgetCard } from './WidgetCard';
import { Lightbulb, ArrowUpRight, ArrowDownRight, Fingerprint } from 'lucide-react';
import { ShapWaterfall } from '../../../../features/explainability/components/ShapWaterfall';

interface Props {
  data: any;
}

export function EnhancedShapPanel({ data }: Props) {
  if (!data?.explanation) return null;

  const isNormal = data.prediction.label !== 'seizure';
  const features = data.explanation.features || [];
  
  // Sort features by magnitude
  const sorted = [...features].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const topFeature = sorted[0];
  const topPositive = [...features].sort((a, b) => b.value - a.value).slice(0, 2);
  const topNegative = [...features].sort((a, b) => a.value - b.value).slice(0, 2);

  return (
    <WidgetCard title="Explainable AI Analysis" icon={<Lightbulb size={16} />}>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left: Human Readable Summary */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <div>
            <h4 className="text-lg font-bold text-gray-900 mb-3">What influenced this prediction?</h4>
            <p className="text-sm text-gray-600 mb-4 leading-relaxed">
              The model classified this segment as <span className="font-semibold text-gray-900">{isNormal ? 'Normal Activity' : 'Seizure'}</span> because:
            </p>
            <ul className="space-y-3">
              {topFeature && (
                <li className="flex gap-2 text-sm text-gray-700 items-start">
                  <Fingerprint size={16} className="text-blue-500 shrink-0 mt-0.5" />
                  <span><span className="font-semibold">{topFeature.featureName.replace('Ch0_', '')}</span> was the primary driver, highly atypical for standard background EEG.</span>
                </li>
              )}
              {topPositive.length > 0 && (
                <li className="flex gap-2 text-sm text-gray-700 items-start">
                  <ArrowUpRight size={16} className="text-red-500 shrink-0 mt-0.5" />
                  <span>Elevated {topPositive[0]?.featureName.replace('Ch0_', '')} significantly increased the seizure probability score.</span>
                </li>
              )}
              {topNegative.length > 0 && (
                <li className="flex gap-2 text-sm text-gray-700 items-start">
                  <ArrowDownRight size={16} className="text-emerald-500 shrink-0 mt-0.5" />
                  <span>Decreased {topNegative[0]?.featureName.replace('Ch0_', '')} pulled the prediction toward normal.</span>
                </li>
              )}
            </ul>
          </div>
          
          <div className="p-4 bg-gray-50 rounded-xl border border-gray-100 mt-auto">
            <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">Most Influential Feature</div>
            <div className="text-sm font-semibold text-gray-900">{topFeature?.featureName}</div>
            <div className="text-xs text-gray-500 mt-1 font-mono">SHAP Value: {topFeature?.value > 0 ? '+' : ''}{topFeature?.value.toFixed(4)}</div>
          </div>
        </div>

        {/* Right: Existing SHAP Plot */}
        <div className="lg:col-span-2 bg-gray-50/30 p-4 rounded-xl border border-gray-100 flex items-center justify-center">
          <ShapWaterfall 
            baseValue={data.explanation.baseValue}
            features={features}
            finalProbability={data.prediction.probabilities.seizure}
          />
        </div>

      </div>
    </WidgetCard>
  );
}
