import React from 'react';
import { EegVisualization } from '../../services/api';

interface RealEegMetadataProps {
  readonly visualization: EegVisualization;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return 'Unavailable';
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${(seconds - minutes * 60).toFixed(3)}s`;
}

function formatClock(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(3).padStart(6, '0')}`;
}

export const RealEegMetadata: React.FC<RealEegMetadataProps> = ({ visualization }) => (
  <section className="bg-slate-900 border border-slate-800 rounded-xl p-4 lg:p-5 space-y-4 shadow-lg">
    <div className="flex items-center justify-between gap-3">
      <div>
        <h2 className="text-xs uppercase font-bold tracking-wider text-slate-300">Real EEG metadata</h2>
        <p className="text-[11px] text-slate-500 mt-1">Values below are read from the uploaded EDF and its dataset annotations.</p>
      </div>
      <span className="text-[10px] uppercase font-bold px-2 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
        REAL EEG ANALYSIS
      </span>
    </div>

    <dl className="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-3 text-xs">
      <div><dt className="text-slate-500">Dataset</dt><dd className="text-slate-200 font-mono mt-0.5">{visualization.dataset}</dd></div>
      <div><dt className="text-slate-500">Filename</dt><dd className="text-slate-200 font-mono mt-0.5 break-all">{visualization.fileName}</dd></div>
      <div><dt className="text-slate-500">Patient ID</dt><dd className="text-slate-200 font-mono mt-0.5">{visualization.patientIdentifier ?? 'Patient ID unavailable'}</dd></div>
      <div><dt className="text-slate-500">File size</dt><dd className="text-slate-200 font-mono mt-0.5">{(visualization.fileSizeBytes / 1_048_576).toFixed(2)} MiB</dd></div>
      <div><dt className="text-slate-500">Sampling rate</dt><dd className="text-slate-200 font-mono mt-0.5">{visualization.samplingRate.toFixed(1)} Hz</dd></div>
      <div><dt className="text-slate-500">Duration</dt><dd className="text-slate-200 font-mono mt-0.5">{formatDuration(visualization.durationSeconds)}</dd></div>
      <div><dt className="text-slate-500">EDF channels</dt><dd className="text-slate-200 font-mono mt-0.5">{visualization.totalChannels}</dd></div>
      <div><dt className="text-slate-500">Usable EEG</dt><dd className="text-slate-200 font-mono mt-0.5">{visualization.eegChannelCount}</dd></div>
      <div><dt className="text-slate-500">Annotations</dt><dd className="text-slate-200 mt-0.5">{visualization.hasSeizureAnnotations ? `${visualization.seizures.length} seizure(s)` : visualization.annotationStatus === 'available' ? 'No seizures' : 'Unavailable'}</dd></div>
      <div><dt className="text-slate-500">Reference EEG</dt><dd className="text-slate-200 mt-0.5">{visualization.referenceAvailable ? 'Available' : 'Not available'}</dd></div>
      <div><dt className="text-slate-500">Visualization</dt><dd className="text-slate-200 font-mono mt-0.5">{visualization.visualizationSampleCount} points/channel</dd></div>
      <div><dt className="text-slate-500">Annotation source</dt><dd className="text-slate-200 font-mono mt-0.5">{visualization.annotationSource ?? 'Unavailable'}</dd></div>
    </dl>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
      <div>
        <h3 className="text-[10px] uppercase font-bold tracking-wider text-slate-500">Seizure intervals</h3>
        {visualization.seizures.length === 0 ? (
          <p className="text-slate-500 mt-2">No seizure annotation available for this recording.</p>
        ) : (
          <div className="mt-2 space-y-1.5">
            {visualization.seizures.map((seizure, index) => (
              <div key={`${seizure.startSeconds}-${seizure.endSeconds}`} className="flex flex-wrap gap-x-3 gap-y-1 text-slate-300 font-mono">
                <span className="text-rose-300">Seizure {index + 1}</span>
                <span>START {formatClock(seizure.startSeconds)}</span>
                <span>END {formatClock(seizure.endSeconds)}</span>
                <span>DURATION {seizure.durationSeconds.toFixed(3)} sec</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div>
        <h3 className="text-[10px] uppercase font-bold tracking-wider text-slate-500">Excluded EDF channels</h3>
        {visualization.excludedChannelDetails.length === 0 ? (
          <p className="text-slate-500 mt-2">None</p>
        ) : (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {visualization.excludedChannelDetails.map((channel) => (
              <span key={channel.name} className="px-2 py-1 rounded bg-slate-950 border border-slate-800 text-slate-400 font-mono">
                {channel.name} ({channel.reason})
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  </section>
);
