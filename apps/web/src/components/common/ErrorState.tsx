import React from 'react';
import { AlertOctagon, RotateCcw, FileText } from 'lucide-react';

interface ErrorStateProps {
  readonly message: string;
  readonly code: string;
  readonly onRetry: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({ message, code, onRetry }) => {
  return (
    <main className="min-h-[75vh] flex items-center justify-center p-6">
      <div className="max-w-md w-full bg-slate-900 border border-rose-900/60 rounded-2xl p-8 shadow-2xl text-center space-y-6">
        <div className="h-16 w-16 bg-rose-950/80 border border-rose-700/50 rounded-2xl flex items-center justify-center mx-auto text-rose-400 shadow-lg shadow-rose-950/50">
          <AlertOctagon className="h-8 w-8 animate-bounce" />
        </div>

        <div className="space-y-2">
          <h2 className="text-xl font-bold text-white">Telemetry Stream Interrupted</h2>
          <p className="text-sm text-slate-400">{message}</p>
          <div className="inline-block mt-2 px-3 py-1 bg-slate-950 rounded text-xs font-mono text-rose-400 border border-slate-800">
            Error Code: {code}
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 pt-2">
          <button
            type="button"
            onClick={onRetry}
            className="flex-1 px-4 py-2.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold text-sm flex items-center justify-center space-x-2 transition shadow-md shadow-rose-900/30"
          >
            <RotateCcw className="h-4 w-4" />
            <span>Reconnect Telemetry</span>
          </button>
          <button
            type="button"
            onClick={() => {
              alert(`Diagnostic Trace: [${code}] Socket handshake failure at 256Hz buffer allocation. Buffer overflow protection engaged.`);
            }}
            className="px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium text-sm flex items-center justify-center space-x-2 transition border border-slate-700"
          >
            <FileText className="h-4 w-4" />
            <span>View Logs</span>
          </button>
        </div>
      </div>
    </main>
  );
};
