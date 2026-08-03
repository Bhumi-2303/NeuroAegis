import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ZoomIn, ZoomOut, Maximize2, Move, Download, SlidersHorizontal, Settings2 } from 'lucide-react';

interface Props {
  data: number[];
}

export function EEGViewer({ data }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Map raw data array to Recharts format
  const chartData = data.map((val, i) => ({ time: i, amplitude: val }));

  return (
    <WidgetCard 
      className={isExpanded ? "fixed inset-4 z-50 shadow-2xl" : "h-[400px]"}
      title="Interactive EEG Viewer"
      icon={<SlidersHorizontal size={16} />}
      headerRight={
        <div className="flex items-center gap-1">
          <ToolbarButton icon={<ZoomIn size={14} />} tooltip="Zoom In" />
          <ToolbarButton icon={<ZoomOut size={14} />} tooltip="Zoom Out" />
          <ToolbarButton icon={<Move size={14} />} tooltip="Pan Tool" />
          <div className="w-px h-4 bg-gray-200 mx-1" />
          <ToolbarButton icon={<Settings2 size={14} />} tooltip="Channel Settings" />
          <ToolbarButton icon={<Download size={14} />} tooltip="Export Image" />
          <ToolbarButton 
            icon={<Maximize2 size={14} />} 
            tooltip={isExpanded ? "Minimize" : "Fullscreen"} 
            onClick={() => setIsExpanded(!isExpanded)} 
          />
        </div>
      }
    >
      <div className="flex-1 w-full bg-white relative">
        {/* Amplitude/Time Scale Watermarks */}
        <div className="absolute top-2 left-2 text-[10px] font-mono text-gray-400 bg-white/80 px-1 rounded">50 µV/div</div>
        <div className="absolute bottom-6 right-4 text-[10px] font-mono text-gray-400 bg-white/80 px-1 rounded">1 sec/div</div>

        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 20, right: 0, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={true} horizontal={true} />
            <XAxis dataKey="time" hide />
            <YAxis domain={['auto', 'auto']} hide />
            <Tooltip 
              contentStyle={{ borderRadius: '8px', border: '1px solid #f0f0f0', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' }}
              labelStyle={{ display: 'none' }}
              itemStyle={{ color: '#2563EB', fontSize: '12px', fontFamily: 'monospace' }}
              formatter={(val: number) => [`${val.toFixed(2)} µV`, 'Amplitude']}
            />
            <Line 
              type="step" 
              dataKey="amplitude" 
              stroke="#111827" 
              strokeWidth={1} 
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </WidgetCard>
  );
}

function ToolbarButton({ icon, tooltip, onClick }: { icon: React.ReactNode, tooltip: string, onClick?: () => void }) {
  return (
    <button 
      onClick={onClick}
      title={tooltip}
      className="p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-900 rounded-md transition-colors"
    >
      {icon}
    </button>
  );
}
