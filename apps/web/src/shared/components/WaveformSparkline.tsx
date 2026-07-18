import { forwardRef, useId } from 'react';
import type { HTMLAttributes } from 'react';

export interface WaveformSparklineProps extends HTMLAttributes<HTMLDivElement> {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  className?: string;
}

export const WaveformSparkline = forwardRef<HTMLDivElement, WaveformSparklineProps>(
  ({ data, width = 120, height = 32, color = 'var(--accent-primary)', className = '', ...props }, ref) => {
    const gradientId = useId();

    if (!data || data.length === 0) {
      return <div ref={ref} className={className} style={{ width, height }} {...props} />;
    }

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    const points = data.map((val, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((val - min) / range) * height;
      return `${x},${y}`;
    }).join(' ');

    const fillPoints = `0,${height} ${points} ${width},${height}`;

    return (
      <div ref={ref} className={`inline-flex ${className}`} {...props}>
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.1} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <polygon points={fillPoints} fill={`url(#${gradientId})`} />
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    );
  }
);

WaveformSparkline.displayName = 'WaveformSparkline';
