import React from 'react';
import { Database, PlusCircle, Play } from 'lucide-react';

interface EmptyStateProps {
  readonly onSelectSamplePatient: () => void;
  readonly onUploadSession: () => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  onSelectSamplePatient,
  onUploadSession,
}) => {
  return (
    <main className="min-h-[75vh] flex items-center justify-center p-6">
      <div className="max-w-lg w-full bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center space-y-6 shadow-xl">
        <div className="h-16 w-16 bg-indigo-950/60 border border-indigo-800/40 rounded-2xl flex items-center justify-center mx-auto text-indigo-400 shadow-md">
          <Database className="h-8 w-8" />
        </div>

        <div className="space-y-2">
          <h2 className="text-xl font-bold text-white">No Active EEG Session Connected</h2>
          <p className="text-sm text-slate-400">
            Select a benchmark clinical recording or connect an ICU patient stream to initiate real-time 16-channel seizure prediction.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
          <button
            type="button"
            onClick={onSelectSamplePatient}
            className="px-4 py-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm flex items-center justify-center space-x-2 transition shadow-lg shadow-indigo-600/30"
          >
            <Play className="h-4 w-4 fill-white" />
            <span>Load CHB-MIT-01</span>
          </button>
          <button
            type="button"
            onClick={onUploadSession}
            className="px-4 py-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-sm flex items-center justify-center space-x-2 transition border border-slate-700"
          >
            <PlusCircle className="h-4 w-4" />
            <span>Upload .EDF Session</span>
          </button>
        </div>
      </div>
    </main>
  );
};
