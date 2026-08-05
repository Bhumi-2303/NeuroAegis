import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ZoomIn, ZoomOut, Maximize2, Move, Download, SlidersHorizontal, Settings2, Lock } from 'lucide-react';

interface Props {
  data: number[];
}

export function EEGViewer({ data }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Map raw data array to Recharts format
  const chartData = data.map((val, i) => ({ time: i, amplitude: val }));

  return (
    <WidgetCard 
      className="h-[400px]"
      title="Interactive EEG Viewer"
      icon={<SlidersHorizontal size={16} />}
    >
      <div className="flex-1 w-full bg-transparent relative flex flex-col m-2 mt-4">
        <div className="flex-1 w-full min-h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
              <XAxis 
                dataKey="time" 
                tick={{ fontSize: 10, fill: '#6B7280' }}
                stroke="#D1D5DB"
                tickCount={10}
              />
              <YAxis 
                domain={['auto', 'auto']}
                tick={{ fontSize: 10, fill: '#6B7280' }}
                stroke="#D1D5DB"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  borderRadius: '8px',
                  border: '1px solid #E5E7EB',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                }}
                labelStyle={{ color: '#374151', fontWeight: 500, marginBottom: '4px' }}
              />
              <Line
                type="monotone"
                dataKey="amplitude"
                stroke="var(--accent-primary)"
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </WidgetCard>
  );
}

function ToolbarButton({ icon, tooltip, onClick }: { icon: React.ReactNode, tooltip: string, onClick?: () => void }) {
  return (
    <button 
      onClick={onClick}
      title={tooltip}
      className="p-1.5 text-[var(--text-secondary)] hover:bg-[var(--bg-3)] hover:text-[var(--text-primary)] rounded-md transition-colors"
    >
      {icon}
    </button>
  );
}
