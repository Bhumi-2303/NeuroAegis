import React, { useEffect } from 'react';
import { PatientVitals } from '../../types/neuroaegis';
import { X, User, Activity, Pill, Stethoscope } from 'lucide-react';

interface PatientDetailModalProps {
  readonly isOpen: boolean;
  readonly vitals: PatientVitals;
  readonly onClose: () => void;
}

export const PatientDetailModal: React.FC<PatientDetailModalProps> = ({
  isOpen,
  vitals,
  onClose,
}) => {
  // Close on Escape key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm"
    >
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full overflow-hidden shadow-2xl space-y-6">
        {/* Modal Header */}
        <div className="bg-slate-950 px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
              <User className="h-4 w-4" />
            </div>
            <div>
              <h2 id="modal-title" className="text-sm font-bold text-white">
                Clinical Patient Profile
              </h2>
              <span className="text-[11px] text-slate-400 font-mono">{vitals.patientId}</span>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close modal"
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-5 text-xs">
          {/* Demographics & Vitals */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-950/50 p-3.5 rounded-xl border border-slate-800/80">
            <div>
              <span className="text-slate-500 uppercase text-[9px] font-bold block">Age / Sex</span>
              <span className="text-slate-200 font-medium">{vitals.age} yo, {vitals.gender}</span>
            </div>
            <div>
              <span className="text-slate-500 uppercase text-[9px] font-bold block">Sampling Rate</span>
              <span className="text-slate-200 font-mono font-medium">{vitals.samplingRateHz} Hz</span>
            </div>
            <div>
              <span className="text-slate-500 uppercase text-[9px] font-bold block">Heart Rate</span>
              <span className="text-emerald-400 font-mono font-bold">{vitals.heartRateBpm} BPM</span>
            </div>
            <div>
              <span className="text-slate-500 uppercase text-[9px] font-bold block">SpO2</span>
              <span className="text-cyan-400 font-mono font-bold">{vitals.spO2Percentage}%</span>
            </div>
          </div>

          {/* Clinical Diagnosis */}
          <div className="space-y-1.5">
            <div className="flex items-center space-x-1.5 text-slate-300 font-semibold">
              <Stethoscope className="h-3.5 w-3.5 text-indigo-400" />
              <span>Primary Neurological Diagnosis</span>
            </div>
            <p className="text-slate-300 bg-slate-950 p-3 rounded-lg border border-slate-800 leading-relaxed">
              {vitals.clinicalDiagnosis}
            </p>
          </div>

          {/* Active Anticonvulsant Medications */}
          <div className="space-y-1.5">
            <div className="flex items-center space-x-1.5 text-slate-300 font-semibold">
              <Pill className="h-3.5 w-3.5 text-indigo-400" />
              <span>Active Anticonvulsant Regimen</span>
            </div>
            <ul className="space-y-1 bg-slate-950 p-3 rounded-lg border border-slate-800 text-slate-300">
              {vitals.activeMedications.map((med) => (
                <li key={med} className="flex items-center space-x-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
                  <span>{med}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Electrode Montage */}
          <div className="space-y-1.5">
            <div className="flex items-center space-x-1.5 text-slate-300 font-semibold">
              <Activity className="h-3.5 w-3.5 text-indigo-400" />
              <span>Montage Configuration</span>
            </div>
            <p className="text-slate-400 bg-slate-950 p-2.5 rounded-lg border border-slate-800 font-mono text-[11px]">
              {vitals.electrodeMontage}
            </p>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="bg-slate-950 px-6 py-3.5 border-t border-slate-800 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs shadow-md transition"
          >
            Dismiss Profile
          </button>
        </div>
      </div>
    </div>
  );
};
