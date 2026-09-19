import React from 'react';
import { ModelThresholdConfig } from '../../types/neuroaegis';
import { Sliders } from 'lucide-react';

interface ThresholdControlPanelProps {
  readonly config: ModelThresholdConfig;
  readonly onChange: (newConfig: ModelThresholdConfig) => void;
}

export const ThresholdControlPanel: React.FC<ThresholdControlPanelProps> = ({
  config,
  onChange,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 lg:p-5 space-y-4 shadow-lg">
      <div className="flex items-center space-x-2">
        <Sliders className="h-4 w-4 text-indigo-400" />
        <h3 className="text-xs uppercase font-bold tracking-wider text-slate-400">
          Inference & DSP Controls
        </h3>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
        {/* Sensitivity Slider */}
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <label htmlFor="sens-slider" className="text-slate-300 font-medium">
              Detection Threshold
            </label>
            <span className="font-mono text-indigo-400 font-bold">
              {(config.sensitivityThreshold * 100).toFixed(0)}%
            </span>
          </div>
          <input
            id="sens-slider"
            type="range"
            min="0.10"
            max="0.95"
            step="0.05"
            value={config.sensitivityThreshold}
            onChange={(e) =>
              onChange({
                ...config,
                sensitivityThreshold: parseFloat(e.target.value),
              })
            }
            className="w-full accent-indigo-500 bg-slate-800 rounded h-1.5 cursor-pointer"
          />
        </div>

        {/* Temporal Smoothing Window Slider */}
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <label htmlFor="temp-window-slider" className="text-slate-300 font-medium">
              Temporal Window
            </label>
            <span className="font-mono text-indigo-400 font-bold">
              {config.temporalWindowSeconds}s
            </span>
          </div>
          <input
            id="temp-window-slider"
            type="range"
            min="1"
            max="8"
            step="1"
            value={config.temporalWindowSeconds}
            onChange={(e) =>
              onChange({
                ...config,
                temporalWindowSeconds: parseInt(e.target.value, 10),
              })
            }
            className="w-full accent-indigo-500 bg-slate-800 rounded h-1.5 cursor-pointer"
          />
        </div>

        {/* Display Gain Multiplier */}
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <label htmlFor="gain-slider" className="text-slate-300 font-medium">
              Amplitude Gain
            </label>
            <span className="font-mono text-indigo-400 font-bold">
              {config.gainMultiplier.toFixed(1)}x
            </span>
          </div>
          <input
            id="gain-slider"
            type="range"
            min="0.5"
            max="3.0"
            step="0.25"
            value={config.gainMultiplier}
            onChange={(e) =>
              onChange({
                ...config,
                gainMultiplier: parseFloat(e.target.value),
              })
            }
            className="w-full accent-indigo-500 bg-slate-800 rounded h-1.5 cursor-pointer"
          />
        </div>
      </div>
    </div>
  );
};
