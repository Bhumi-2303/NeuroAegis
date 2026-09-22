import React from 'react';
import { EegChannelActivity } from '../../services/api';

interface SignalDerivedChannelActivityProps {
  readonly activity: readonly EegChannelActivity[];
  readonly available: boolean;
  readonly note: string;
}

export const SignalDerivedChannelActivity: React.FC<SignalDerivedChannelActivityProps> = ({
  activity,
  available,
  note,
}) => (
  <section className="bg-slate-900 border border-slate-800 rounded-xl p-4 lg:p-5 space-y-3 shadow-lg">
    <div>
      <h2 className="text-xs uppercase font-bold tracking-wider text-slate-300">Signal-derived channel activity</h2>
      <p className="text-[11px] text-slate-500 mt-1">This score is calculated from EEG signal characteristics during the annotated seizure interval. It is not a channel-level model prediction.</p>
    </div>
    {!available || activity.length === 0 ? (
      <p className="text-xs text-slate-500 py-3">{note}</p>
    ) : (
      <div className="max-h-64 overflow-y-auto space-y-1.5 pr-1">
        {activity.map((item) => (
          <div key={item.channelName} className="grid grid-cols-[minmax(0,1fr)_auto_auto] gap-3 items-center text-xs border-b border-slate-800/70 pb-1.5">
            <span className="text-slate-300 font-mono truncate">{item.channelName}</span>
            <span className="text-slate-400 font-mono">score {item.activityScore.toFixed(3)}</span>
            <span className={item.relativeChange >= 0 ? 'text-rose-300 font-mono' : 'text-cyan-300 font-mono'}>
              {item.relativeChange >= 0 ? '+' : ''}{(item.relativeChange * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    )}
  </section>
);
