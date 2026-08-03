import { WidgetCard } from './WidgetCard';
import { BarChart3 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export function GlobalFeatureImportance() {
  // Mock data for top 15 features across the dataset
  const data = [
    { name: 'Variance', value: 0.18 },
    { name: 'Energy', value: 0.15 },
    { name: 'Line Length', value: 0.12 },
    { name: 'Wavelet Energy', value: 0.11 },
    { name: 'Bandpower Delta', value: 0.09 },
    { name: 'Hjorth Complexity', value: 0.08 },
    { name: 'Shannon Entropy', value: 0.07 },
    { name: 'Bandpower Theta', value: 0.05 },
    { name: 'Mean Amplitude', value: 0.04 },
    { name: 'Zero Crossing', value: 0.03 },
    { name: 'Kurtosis', value: 0.02 },
    { name: 'Skewness', value: 0.02 },
    { name: 'Peak Frequency', value: 0.015 },
    { name: 'Spectral Edge', value: 0.015 },
    { name: 'Hjorth Mobility', value: 0.01 },
  ].sort((a, b) => b.value - a.value);

  return (
    <WidgetCard title="Global Feature Importance" icon={<BarChart3 size={16} />}>
      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 20, left: 20, bottom: 0 }}>
            <XAxis type="number" hide />
            <YAxis 
              dataKey="name" 
              type="category" 
              axisLine={false} 
              tickLine={false}
              tick={{ fontSize: 11, fill: '#6B7280' }}
              width={120}
            />
            <Tooltip 
              cursor={{ fill: '#F3F4F6' }}
              contentStyle={{ borderRadius: '8px', border: '1px solid #f0f0f0', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' }}
              itemStyle={{ color: '#111827', fontSize: '12px' }}
              formatter={(val: any) => [`${(Number(val || 0) * 100).toFixed(1)}%`, 'Relative Importance']}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={12}>
              {data.map((_entry, index) => (
                <Cell key={`cell-${index}`} fill={index < 3 ? '#2563EB' : '#93C5FD'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </WidgetCard>
  );
}
