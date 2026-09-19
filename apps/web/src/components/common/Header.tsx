import React from 'react';
import { Activity, Download, AlertCircle, RefreshCw, Volume2, VolumeX, User } from 'lucide-react';
import { PatientVitals } from '../../types/neuroaegis';

interface HeaderProps {
  readonly vitals: PatientVitals;
  readonly isStreaming: boolean;
  readonly audioEnabled: boolean;
  readonly onToggleStream: () => void;
  readonly onToggleAudio: () => void;
  readonly onExportData: (format: 'json' | 'csv') => void;
  readonly onTriggerManualSeizure: () => void;
  readonly isManualSeizureActive: boolean;
  readonly onOpenPatientModal: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  vitals,
  isStreaming,
  audioEnabled,
  onToggleStream,
  onToggleAudio,
  onExportData,
  onTriggerManualSeizure,
  isManualSeizureActive,
  onOpenPatientModal,
}) => {
  return (
    <header className="bg-slate-900/95 backdrop-blur border-b border-slate-800 px-4 lg:px-6 py-3.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30 shadow-md">
      {/* Brand & Status */}
      <div className="flex items-center space-x-3">
        <div className="h-9 w-9 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/25">
          <Activity className="h-5 w-5 text-white" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-base lg:text-lg font-bold text-white tracking-wide">
              NEURO<span className="text-indigo-400">AEGIS</span>
            </h1>
            <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-700/50">
              v2.4 ICU
            </span>
          </div>
          <p className="text-[11px] text-slate-400">Real-Time EEG & Seizure Telemetry</p>
        </div>

        <div className="hidden sm:flex items-center space-x-2 pl-3 border-l border-slate-800 text-xs">
          <span className="flex h-2.5 w-2.5 relative">
            <span
              className={`animate-ping absolute inline-flex h-full w-full rounded-full ${
                isStreaming ? 'bg-emerald-400 opacity-75' : 'bg-amber-400 opacity-75'
              }`}
            />
            <span
              className={`relative inline-flex rounded-full h-2.5 w-2.5 ${
                isStreaming ? 'bg-emerald-500' : 'bg-amber-500'
              }`}
            />
          </span>
          <span className="text-slate-300 font-medium">
            {isStreaming ? 'LIVE 256Hz' : 'PAUSED'}
          </span>
        </div>
      </div>

      {/* Patient Vital Stats Header Strip */}
      <div className="hidden lg:flex items-center space-x-5 bg-slate-950/80 border border-slate-800 rounded-lg px-3.5 py-1.5 text-xs">
        <button
          type="button"
          onClick={onOpenPatientModal}
          className="flex items-center space-x-2 text-left hover:text-indigo-300 transition group"
        >
          <User className="h-3.5 w-3.5 text-slate-400 group-hover:text-indigo-400" />
          <div>
            <span className="text-slate-500 uppercase text-[9px] block font-bold tracking-wider">
              Patient
            </span>
            <span className="text-slate-200 font-mono font-medium group-hover:underline">
              {vitals.patientId}
            </span>
          </div>
        </button>
        <div className="h-5 w-px bg-slate-800" />
        <div>
          <span className="text-slate-500 uppercase text-[9px] block font-bold tracking-wider">
            Heart Rate
          </span>
          <span className="text-emerald-400 font-mono font-semibold">
            {vitals.heartRateBpm} BPM
          </span>
        </div>
        <div className="h-5 w-px bg-slate-800" />
        <div>
          <span className="text-slate-500 uppercase text-[9px] block font-bold tracking-wider">
            SpO2
          </span>
          <span className="text-cyan-400 font-mono font-semibold">
            {vitals.spO2Percentage}%
          </span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center space-x-2">
        <button
          type="button"
          onClick={onTriggerManualSeizure}
          aria-label="Simulate Epileptic Seizure Pattern"
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition-colors border ${
            isManualSeizureActive
              ? 'bg-rose-600 border-rose-500 text-white animate-pulse'
              : 'bg-slate-800 border-slate-700 text-rose-300 hover:bg-rose-950/40 hover:border-rose-800'
          }`}
        >
          <AlertCircle className="h-3.5 w-3.5" />
          <span>{isManualSeizureActive ? 'Stop Seizure Sim' : 'Inject Seizure Sim'}</span>
        </button>

        <button
          type="button"
          onClick={onToggleStream}
          aria-label={isStreaming ? 'Pause streaming telemetry' : 'Resume streaming telemetry'}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition-colors border ${
            isStreaming
              ? 'bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-700'
              : 'bg-emerald-600 border-emerald-500 text-white hover:bg-emerald-500'
          }`}
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isStreaming ? 'animate-spin' : ''}`} />
          <span>{isStreaming ? 'Pause' : 'Resume'}</span>
        </button>

        <button
          type="button"
          onClick={onToggleAudio}
          aria-label={audioEnabled ? 'Mute audio alarms' : 'Unmute audio alarms'}
          className="p-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:text-white hover:bg-slate-700 transition"
        >
          {audioEnabled ? (
            <Volume2 className="h-4 w-4 text-emerald-400" />
          ) : (
            <VolumeX className="h-4 w-4 text-slate-500" />
          )}
        </button>

        <div className="relative group">
          <button
            type="button"
            aria-label="Export Telemetry Session Data"
            className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium flex items-center space-x-1.5 shadow-sm transition"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export</span>
          </button>
          <div className="absolute right-0 mt-1 w-32 bg-slate-900 border border-slate-800 rounded-md shadow-xl py-1 hidden group-hover:block z-50">
            <button
              type="button"
              onClick={() => onExportData('json')}
              className="w-full text-left px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800 hover:text-white"
            >
              Export JSON
            </button>
            <button
              type="button"
              onClick={() => onExportData('csv')}
              className="w-full text-left px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800 hover:text-white"
            >
              Export CSV
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
