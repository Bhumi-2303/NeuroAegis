import { WidgetCard } from './WidgetCard';
import { BarChart3 } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface FeatureImportance {
  name: string;
  value: number;
  category: string;
}



export function GlobalFeatureImportance() {
  const { data, isLoading, error } = useQuery<{ feature_importances: FeatureImportance[] }>({
    queryKey: ['model-info'],
    queryFn: async () => {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await fetch(`${API_URL}/api/v1/model/info?dataset=bonn&model_name=lightgbm`);
      if (!res.ok) throw new Error('Failed to fetch model info');
      return res.json();
    },
    staleTime: Infinity,
  });

  const chartData = data?.feature_importances
    ? [...data.feature_importances]
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .filter(f => Math.abs(f.value) > 0)
        .slice(0, 15)
    : [];
  
  // A color mapping for the simplistic heuristic categories we made
  const categoryColors: Record<string, string> = {
    Temporal: '#8b5cf6', // Violet
    Frequency: '#3b82f6', // Blue
    Wavelet: '#ec4899', // Pink
    Entropy: '#f59e0b', // Amber
    Hjorth: '#10b981', // Emerald
  };

  return (
    <WidgetCard title="Global Feature Importance (LightGBM)" icon={<BarChart3 size={16} />}>
      {isLoading ? (
        <div className="flex items-center justify-center h-[300px] text-[var(--text-secondary)]">Loading model weights...</div>
      ) : error ? (
        <div className="flex items-center justify-center h-[300px] text-red-500">Failed to load importances</div>
      ) : (
        <div className="h-[300px] w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
              <XAxis type="number" hide />
              <YAxis 
                type="category" 
                dataKey="name" 
                width={120} 
                tick={{ fontSize: 10, fill: '#6b7280' }} 
                axisLine={false} 
                tickLine={false} 
              />
              <Tooltip 
                cursor={{ fill: '#f3f4f6' }}
                contentStyle={{ borderRadius: '8px', border: '1px solid #f0f0f0', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' }}
                formatter={(val: any, _name: any, props: any) => [
                  `${Number(val || 0).toFixed(2)}%`,
                  props.payload.category || 'Importance'
                ]}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={12}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={categoryColors[entry.category] || '#3b82f6'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </WidgetCard>
  );
}
