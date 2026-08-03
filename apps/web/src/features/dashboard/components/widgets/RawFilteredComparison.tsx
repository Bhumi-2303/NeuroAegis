import React from 'react';
import { WidgetCard } from './WidgetCard';
import { Activity } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';

interface Props {
  data: number[];
}

export function RawFilteredComparison({ data }: Props) {
  // Generate mock filtered data by smoothing the raw data
  const filteredData = data.map((val, i) => {
    // simple moving average mock
    const prev = data[i - 1] || val;
    const next = data[i + 1] || val;
    return {
      time: i,
      raw: val,
      filtered: (val + prev + next) / 3
    };
  });

  return (
    <WidgetCard title="Preprocessing Comparison" icon={<Activity size={16} />}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-[250px]">
        {/* Left: Raw */}
        <div className="flex flex-col border border-gray-100 rounded-xl overflow-hidden bg-gray-50/30">
           <div className="px-3 py-2 border-b border-gray-100 flex justify-between items-center bg-white">
             <span className="text-xs font-semibold text-gray-700">Raw Signal</span>
             <span className="text-[10px] text-gray-400 font-mono">Unfiltered</span>
           </div>
           <div className="flex-1 w-full p-2">
             <ResponsiveContainer width="100%" height="100%">
               <LineChart data={filteredData}>
                 <YAxis domain={['auto', 'auto']} hide />
                 <Line type="monotone" dataKey="raw" stroke="#ef4444" strokeWidth={1} dot={false} isAnimationActive={false} />
               </LineChart>
             </ResponsiveContainer>
           </div>
        </div>

        {/* Right: Filtered */}
        <div className="flex flex-col border border-gray-100 rounded-xl overflow-hidden bg-gray-50/30">
           <div className="px-3 py-2 border-b border-gray-100 flex justify-between items-center bg-white">
             <span className="text-xs font-semibold text-gray-700">Filtered Signal</span>
             <span className="text-[10px] text-gray-400 font-mono">Bandpass + Notch</span>
           </div>
           <div className="flex-1 w-full p-2">
             <ResponsiveContainer width="100%" height="100%">
               <LineChart data={filteredData}>
                 <YAxis domain={['auto', 'auto']} hide />
                 <Line type="monotone" dataKey="filtered" stroke="#2563eb" strokeWidth={1.5} dot={false} isAnimationActive={false} />
               </LineChart>
             </ResponsiveContainer>
           </div>
        </div>
      </div>
    </WidgetCard>
  );
}
