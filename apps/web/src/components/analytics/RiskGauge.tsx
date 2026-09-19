import React from 'react';
import { ModelPrediction } from '../../types/neuroaegis';
import { ShieldAlert, CheckCircle2 } from 'lucide-react';

interface RiskGaugeProps {
  readonly prediction: ModelPrediction;
  readonly sensitivityThreshold: number;
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({
  prediction,
  sensitivityThreshold,
}) => {
  const seizureProb = prediction.probabilities.seizure;
  const percentage = Math.round(seizureProb * 100);
  const isAlarm = seizureProb >= sensitivityThreshold;

  // SVG Gauge arc math
  const radius = 58;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - seizureProb * circumference * 0.75;

  const colorClass = isAlarm
    ? 'text-rose-500 stroke-rose-500'
    : seizureProb > 0.3
    ? 'text-amber-400 stroke-amber-400'
    : 'text-emerald-400 stroke-emerald-400';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 lg:p-5 space-y-4 shadow-lg">
      <div className="flex items-center justify-between">
        <h3 className="text-xs uppercase font-bold tracking-wider text-slate-400">
          Seizure Probability (AI)
        </h3>
        <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono font-medium">
          {prediction.modelName.toUpperCase()}
        </span>
      </div>

      <div className="flex flex-col items-center justify-center py-1 relative">
        <svg className="w-32 h-32 transform -rotate-135">
          {/* Background circle */}
          <circle
            cx="64"
            cy="64"
            r={radius}
            stroke="currentColor"
            strokeWidth="9"
            className="text-slate-800"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * 0.25}
          />
          {/* Active progress arc */}
          <circle
            cx="64"
            cy="64"
            r={radius}
            stroke="currentColor"
            strokeWidth="9"
            className={`${colorClass} transition-all duration-300`}
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <span className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
            {percentage}%
          </span>
          <span className="text-[10px] uppercase font-semibold text-slate-400">
            {isAlarm ? 'Critical Risk' : 'Normal Baseline'}
          </span>
        </div>
      </div>

      <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-xs">
        <div className="flex items-center space-x-1.5">
          {isAlarm ? (
            <ShieldAlert className="h-4 w-4 text-rose-500" />
          ) : (
            <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          )}
          <span className="text-slate-300 font-medium">
            Confidence: <span className="uppercase text-white font-bold">{prediction.confidence.band}</span> ({(prediction.confidence.value * 100).toFixed(0)}%)
          </span>
        </div>
        <span className="text-slate-500 font-mono text-[11px]">
          Threshold: {(sensitivityThreshold * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
};
