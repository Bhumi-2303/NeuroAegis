import React from 'react';
import { SeizureAlert } from '../../types/neuroaegis';
import { Bell, Check, ShieldCheck } from 'lucide-react';

interface AlertDrawerProps {
  readonly alerts: readonly SeizureAlert[];
  readonly onAcknowledgeAlert: (id: string) => void;
  readonly onClearAll: () => void;
}

export const AlertDrawer: React.FC<AlertDrawerProps> = ({
  alerts,
  onAcknowledgeAlert,
  onClearAll,
}) => {
  const unacknowledgedCount = alerts.filter((a) => !a.acknowledged).length;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 lg:p-5 space-y-4 shadow-lg">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Bell className="h-4 w-4 text-slate-400" />
          <h3 className="text-xs uppercase font-bold tracking-wider text-slate-400">
            Clinical Event Logs
          </h3>
          {unacknowledgedCount > 0 && (
            <span className="px-1.5 py-0.5 rounded-full bg-rose-600 text-white text-[10px] font-bold animate-pulse">
              {unacknowledgedCount} NEW
            </span>
          )}
        </div>
        {alerts.length > 0 && (
          <button
            type="button"
            onClick={onClearAll}
            className="text-[10px] text-slate-400 hover:text-slate-200 transition underline underline-offset-2"
          >
            Clear History
          </button>
        )}
      </div>

      <div className="space-y-2.5 max-h-52 overflow-y-auto pr-1">
        {alerts.length === 0 ? (
          <div className="text-center py-6 text-slate-500 text-xs flex flex-col items-center space-y-1">
            <ShieldCheck className="h-6 w-6 text-slate-600" />
            <span>No anomalous discharges recorded</span>
          </div>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.id}
              className={`p-3 rounded-lg border text-xs flex items-center justify-between transition ${
                alert.acknowledged
                  ? 'bg-slate-950/60 border-slate-800/80 opacity-70'
                  : 'bg-rose-950/30 border-rose-800/60'
              }`}
            >
              <div className="space-y-0.5">
                <div className="flex items-center space-x-2">
                  <span
                    className={`h-2 w-2 rounded-full ${
                      alert.severity === 'critical' ? 'bg-rose-500' : 'bg-amber-400'
                    }`}
                  />
                  <span className="font-bold text-white uppercase">{alert.primaryChannel}</span>
                  <span className="text-slate-400 text-[10px] font-mono">{alert.timestamp}</span>
                </div>
                <p className="text-[11px] text-slate-300">
                  Paroxysm: <span className="font-semibold text-white">{alert.durationSeconds}s</span> (p=
                  {(alert.probability * 100).toFixed(0)}%)
                </p>
              </div>

              {!alert.acknowledged ? (
                <button
                  type="button"
                  onClick={() => onAcknowledgeAlert(alert.id)}
                  aria-label="Acknowledge Alert"
                  className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[10px] font-medium border border-slate-700 flex items-center space-x-1"
                >
                  <Check className="h-3 w-3 text-emerald-400" />
                  <span>ACK</span>
                </button>
              ) : (
                <span className="text-[10px] text-slate-500 font-mono">ACKED</span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};
