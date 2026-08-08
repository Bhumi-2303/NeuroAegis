import { Info } from 'lucide-react';

/**
 * Displays transparent model limitations alongside any prediction result.
 * Surfaces raw metrics (probability, confidence band) and training data provenance
 * so clinicians can form their own judgement rather than relying on misleading labels.
 */
export function ModelLimitations({ datasetName, modelName }: { datasetName?: string; modelName?: string }) {
  return (
    <div className="bg-sky-500/5 border border-sky-500/20 rounded-xl p-4 mt-4">
      <div className="flex items-start gap-3">
        <Info className="w-5 h-5 text-sky-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-2">
          <p className="font-semibold text-sky-300 text-sm">Model Limitations & Provenance</p>
          <ul className="text-xs text-neutral-400 list-disc list-inside space-y-1">
            <li>
              Trained on <strong className="text-sky-200">{datasetName || 'CHB-MIT / Bonn'}</strong> dataset
              — a public research dataset of pediatric/epilepsy-center recordings. It may not generalize
              to your patient's demographics, recording equipment, or clinical setting.
            </li>
            <li>
              Model: <strong className="text-sky-200">{modelName || 'LightGBM'}</strong>. Reported metrics are from
              held-out test splits of the training data, <em>not</em> independent clinical validation.
            </li>
            <li>
              Confidence values reflect <em>model probability output</em>, not real-world predictive certainty.
              A "High" confidence prediction can still be wrong.
            </li>
            <li>
              EEG artifact rejection, patient movement, and non-standard montages may significantly degrade accuracy.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
