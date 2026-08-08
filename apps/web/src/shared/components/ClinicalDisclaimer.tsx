import { AlertTriangle } from 'lucide-react';

/**
 * A persistent, non-dismissible clinical disclaimer banner.
 * This MUST appear on every page that displays prediction results.
 * It cannot be hidden, minimized, or dismissed by the user.
 */
export function ClinicalDisclaimer() {
  return (
    <div
      role="alert"
      aria-live="polite"
      className="bg-amber-500/10 border-2 border-amber-500/30 rounded-xl p-4 mb-6"
    >
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-6 h-6 text-amber-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-semibold text-amber-300 text-sm uppercase tracking-wide">
            Not a Medical Device — Research Use Only
          </p>
          <p className="text-xs text-neutral-300 leading-relaxed">
            NeuroAegis is an experimental research tool. Its predictions have <strong className="text-amber-200">not</strong> been
            validated in clinical trials and are <strong className="text-amber-200">not</strong> approved by the FDA, EMA, or any
            regulatory body. All outputs must be reviewed and confirmed by a qualified neurologist or
            epileptologist before any clinical action is taken. <strong className="text-amber-200">Never</strong> use these
            results as the sole basis for diagnosis, treatment, or medication changes.
          </p>
        </div>
      </div>
    </div>
  );
}
