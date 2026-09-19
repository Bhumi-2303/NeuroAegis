import React, { useMemo } from 'react';
import { EegChannelData } from '../../types/neuroaegis';

interface EegWaveformCanvasProps {
  readonly channels: readonly EegChannelData[];
  readonly isSeizureActive: boolean;
  readonly gain: number;
}

export const EegWaveformCanvas: React.FC<EegWaveformCanvasProps> = ({
  channels,
  isSeizureActive,
  gain,
}) => {
  const svgWidth = 840;
  const channelHeight = 48;
  const svgHeight = channels.length * channelHeight + 36;

  // Convert voltage buffer into SVG polyline path coordinates
  const channelPaths = useMemo(() => {
    return channels.map((channel, idx) => {
      const sampleCount = channel.voltageMicrovolts.length;
      if (sampleCount === 0) return { name: channel.name, path: '', yBase: 0 };

      const yBase = idx * channelHeight + 30;
      const xStep = svgWidth / (sampleCount - 1);

      const pathString = channel.voltageMicrovolts.reduce((acc, voltage, i) => {
        const x = Number((i * xStep).toFixed(1));
        // Inverse Y: Positive microvolts deflect UPWARD (clinical standard)
        const y = Number((yBase - voltage * 0.35 * gain).toFixed(1));
        return `${acc} ${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      }, '');

      return {
        name: channel.name,
        path: pathString,
        yBase,
      };
    });
  }, [channels, channelHeight, gain, svgWidth]);

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col">
      {/* Waveform Controls Header */}
      <div className="bg-slate-900/90 border-b border-slate-800 px-4 py-2.5 flex items-center justify-between text-xs">
        <div className="flex items-center space-x-3 sm:space-x-4">
          <span className="font-semibold text-slate-200">10-20 Bipolar Longitudinal Montage</span>
          <span className="text-slate-400 hidden sm:inline">30mm/s</span>
          <span className="text-slate-400 font-mono">Gain: {gain.toFixed(1)}x</span>
        </div>
        {isSeizureActive && (
          <span className="flex items-center space-x-1.5 px-2.5 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-700 animate-pulse font-bold tracking-wider text-[11px]">
            <span>●</span>
            <span>ICTAL DISCHARGE ACTIVE</span>
          </span>
        )}
      </div>

      {/* SVG EEG Grid & Signal Traces */}
      <div className="relative overflow-x-auto p-2 bg-slate-950">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full h-auto select-none"
          style={{ minWidth: '680px' }}
        >
          <defs>
            {/* Background Major/Minor Grid Pattern */}
            <pattern id="eegGrid" width="40" height="20" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 20" fill="none" stroke="#1e293b" strokeWidth="0.4" strokeDasharray="1,3" />
            </pattern>
          </defs>

          {/* Grid Background */}
          <rect width={svgWidth} height={svgHeight} fill="url(#eegGrid)" />

          {/* Seizure Region Overlay */}
          {isSeizureActive && (
            <rect
              x={svgWidth * 0.4}
              y={0}
              width={svgWidth * 0.6}
              height={svgHeight}
              fill="rgba(225, 29, 72, 0.07)"
              stroke="rgba(225, 29, 72, 0.3)"
              strokeDasharray="4,4"
            />
          )}

          {/* Channel Waveform Curves & Labels */}
          {channelPaths.map((item) => (
            <g key={`chan-svg-${item.name}`}>
              {/* Channel Label */}
              <text
                x="8"
                y={item.yBase - 6}
                fill="#94a3b8"
                fontSize="9.5"
                fontFamily="ui-monospace, monospace"
                fontWeight="600"
              >
                {item.name}
              </text>

              {/* Zero Voltage Baseline Reference Line */}
              <line
                x1="0"
                y1={item.yBase}
                x2={svgWidth}
                y2={item.yBase}
                stroke="#1e293b"
                strokeWidth="0.5"
                strokeDasharray="2,4"
              />

              {/* High-Resolution EEG Trace */}
              <path
                d={item.path}
                fill="none"
                stroke={isSeizureActive ? '#fb7185' : '#38bdf8'}
                strokeWidth={isSeizureActive ? '1.4' : '1.1'}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
};
