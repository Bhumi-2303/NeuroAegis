import React, { useMemo } from 'react';
import { EegChannelData } from '../../types/neuroaegis';
import { EegSeizureInterval } from '../../services/api';

interface EegWaveformCanvasProps {
  readonly channels: readonly EegChannelData[];
  readonly isSeizureActive?: boolean;
  readonly gain: number;
  readonly samplingRate?: number;
  readonly durationSeconds?: number;
  readonly timeStartSeconds?: number;
  readonly timeEndSeconds?: number;
  readonly seizures?: readonly EegSeizureInterval[];
  readonly title?: string;
  readonly emptyMessage?: string;
}

const SVG_WIDTH = 960;
const LABEL_WIDTH = 112;
const CHANNEL_HEIGHT = 54;
const TOP_PADDING = 24;
const BOTTOM_PADDING = 34;
const MAX_RENDERED_POINTS = 3000;

function formatSeconds(value: number): string {
  if (!Number.isFinite(value)) return '--:--.---';
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${seconds.toFixed(3).padStart(6, '0')}`;
}

function boundedSamples(samples: readonly number[]): number[] {
  if (samples.length <= MAX_RENDERED_POINTS) return [...samples];
  const indices = new Set<number>();
  for (let index = 0; index < MAX_RENDERED_POINTS; index += 1) {
    indices.add(Math.round((index * (samples.length - 1)) / (MAX_RENDERED_POINTS - 1)));
  }
  return [...indices].sort((a, b) => a - b).map((index) => samples[index]);
}

export const EegWaveformCanvas: React.FC<EegWaveformCanvasProps> = ({
  channels,
  isSeizureActive = false,
  gain,
  samplingRate = 256,
  durationSeconds,
  timeStartSeconds = 0,
  timeEndSeconds,
  seizures = [],
  title = 'EEG Waveform',
  emptyMessage = 'No waveform available',
}) => {
  const sampleCount = channels[0]?.voltageMicrovolts.length ?? 0;
  const resolvedDuration = durationSeconds ?? (
    sampleCount > 0 && samplingRate > 0 ? sampleCount / samplingRate : 0
  );
  const resolvedEnd = timeEndSeconds ?? timeStartSeconds + resolvedDuration;
  const timeSpan = Math.max(resolvedEnd - timeStartSeconds, Number.EPSILON);
  const plotWidth = SVG_WIDTH - LABEL_WIDTH;
  const svgHeight = channels.length * CHANNEL_HEIGHT + TOP_PADDING + BOTTOM_PADDING;

  const channelPaths = useMemo(() => channels.map((channel, channelIndex) => {
    const samples = boundedSamples(channel.voltageMicrovolts);
    const yBase = TOP_PADDING + channelIndex * CHANNEL_HEIGHT + CHANNEL_HEIGHT / 2;
    if (samples.length === 0) {
      return { name: channel.name, path: '', yBase };
    }

    const finiteSamples = samples.map((value) => Number.isFinite(value) ? value : 0);
    const center = finiteSamples.reduce((sum, value) => sum + value, 0) / finiteSamples.length;
    const peak = Math.max(...finiteSamples.map((value) => Math.abs(value - center)), 1);
    const scale = (CHANNEL_HEIGHT * 0.38) / peak;
    const xStep = samples.length > 1 ? plotWidth / (samples.length - 1) : plotWidth;
    const path = finiteSamples.map((value, sampleIndex) => {
      const x = LABEL_WIDTH + sampleIndex * xStep;
      const y = yBase - (value - center) * scale * gain;
      return `${sampleIndex === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${Math.max(yBase - 20, Math.min(yBase + 20, y)).toFixed(1)}`;
    }).join(' ');

    return { name: channel.name, path, yBase };
  }), [channels, gain, plotWidth]);

  if (channels.length === 0) {
    return (
      <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
        <div className="bg-slate-900/90 border-b border-slate-800 px-4 py-2.5 text-xs font-semibold text-slate-300">
          {title}
        </div>
        <div className="min-h-32 flex items-center justify-center text-sm text-slate-500">
          {emptyMessage}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col">
      <div className="bg-slate-900/90 border-b border-slate-800 px-4 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-slate-200">{title}</span>
          <span className="text-slate-400">{channels.length} usable channels</span>
          <span className="text-slate-400 font-mono">{samplingRate.toFixed(1)} Hz</span>
          <span className="text-slate-400 font-mono">Gain: {gain.toFixed(1)}x</span>
        </div>
        {(isSeizureActive || seizures.length > 0) && (
          <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-700 font-bold tracking-wider text-[11px]">
            <span>SEIZURE ANNOTATION</span>
          </span>
        )}
      </div>

      <div className="relative overflow-auto max-h-[680px] p-2 bg-slate-950">
        <svg
          viewBox={`0 0 ${SVG_WIDTH} ${svgHeight}`}
          className="w-full h-auto select-none"
          style={{ minWidth: '720px' }}
          role="img"
          aria-label={`${title}, ${channels.length} synchronized EEG channels`}
        >
          <defs>
            <pattern id="eegGrid" width="40" height="20" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 20" fill="none" stroke="#1e293b" strokeWidth="0.4" strokeDasharray="1,3" />
            </pattern>
          </defs>
          <rect width={SVG_WIDTH} height={svgHeight} fill="url(#eegGrid)" />

          {seizures.map((seizure, index) => {
            const start = Math.max(timeStartSeconds, seizure.startSeconds);
            const end = Math.min(resolvedEnd, seizure.endSeconds);
            if (end <= start || resolvedEnd <= timeStartSeconds) return null;
            const x = LABEL_WIDTH + ((start - timeStartSeconds) / timeSpan) * plotWidth;
            const width = ((end - start) / timeSpan) * plotWidth;
            return (
              <g key={`seizure-overlay-${index}`}>
                <rect
                  x={x}
                  y={0}
                  width={Math.max(width, 1)}
                  height={svgHeight - BOTTOM_PADDING}
                  fill="rgba(225, 29, 72, 0.14)"
                  stroke="rgba(251, 113, 133, 0.75)"
                  strokeDasharray="4,4"
                />
                <text x={x + 4} y={14} fill="#fda4af" fontSize="9" fontFamily="ui-monospace, monospace" fontWeight="700">
                  {`SEIZURE ${index + 1}`}
                </text>
              </g>
            );
          })}

          {channelPaths.map((item) => (
            <g key={`chan-svg-${item.name}`}>
              <text x="8" y={item.yBase - 6} fill="#94a3b8" fontSize="9.5" fontFamily="ui-monospace, monospace" fontWeight="600">
                {item.name}
              </text>
              <line x1={LABEL_WIDTH} y1={item.yBase} x2={SVG_WIDTH} y2={item.yBase} stroke="#1e293b" strokeWidth="0.5" strokeDasharray="2,4" />
              <path
                d={item.path}
                fill="none"
                stroke={isSeizureActive ? '#fb7185' : '#38bdf8'}
                strokeWidth={isSeizureActive ? '1.2' : '1.1'}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </g>
          ))}

          <line x1={LABEL_WIDTH} y1={svgHeight - BOTTOM_PADDING + 4} x2={SVG_WIDTH} y2={svgHeight - BOTTOM_PADDING + 4} stroke="#475569" strokeWidth="0.7" />
          <text x={LABEL_WIDTH} y={svgHeight - 7} fill="#64748b" fontSize="9" fontFamily="ui-monospace, monospace">{formatSeconds(timeStartSeconds)}</text>
          <text x={LABEL_WIDTH + plotWidth / 2 - 24} y={svgHeight - 7} fill="#64748b" fontSize="9" fontFamily="ui-monospace, monospace">{formatSeconds((timeStartSeconds + resolvedEnd) / 2)}</text>
          <text x={SVG_WIDTH - 58} y={svgHeight - 7} fill="#64748b" fontSize="9" fontFamily="ui-monospace, monospace">{formatSeconds(resolvedEnd)}</text>
        </svg>
      </div>
    </div>
  );
};
