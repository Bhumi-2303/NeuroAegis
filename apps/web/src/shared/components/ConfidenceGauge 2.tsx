import { forwardRef } from 'react';
import type { HTMLAttributes } from 'react';
import { motion } from 'framer-motion';

export interface ConfidenceGaugeProps extends HTMLAttributes<HTMLDivElement> {
  value: number;
  size?: number;
  label?: string;
  className?: string;
}

export const ConfidenceGauge = forwardRef<HTMLDivElement, ConfidenceGaugeProps>(
  ({ value, size = 120, label, className = '', ...props }, ref) => {
    const clampedValue = Math.max(0, Math.min(100, value));
    
    let color = 'var(--state-success)';
    if (clampedValue < 40) color = 'var(--state-danger)';
    else if (clampedValue < 70) color = 'var(--state-warning)';

    const strokeWidth = size * 0.08;
    const radius = (size - strokeWidth) / 2;
    const center = size / 2;
    
    // 270 degree arc
    const circumference = 2 * Math.PI * radius;
    const arcLength = circumference * 0.75; 
    const dashArray = `${arcLength} ${circumference}`;
    const dashOffset = arcLength - (clampedValue / 100) * arcLength;

    return (
      <div ref={ref} className={`flex flex-col items-center justify-center ${className}`} {...props}>
        <div className="relative" style={{ width: size, height: size }}>
          <svg width={size} height={size} style={{ transform: 'rotate(135deg)' }}>
            <circle
              cx={center}
              cy={center}
              r={radius}
              fill="none"
              stroke="rgba(255,255,255,0.06)"
              strokeWidth={strokeWidth}
              strokeDasharray={dashArray}
              strokeLinecap="round"
            />
            <motion.circle
              cx={center}
              cy={center}
              r={radius}
              fill="none"
              stroke={color}
              strokeWidth={strokeWidth}
              strokeDasharray={dashArray}
              strokeLinecap="round"
              initial={{ strokeDashoffset: arcLength }}
              animate={{ strokeDashoffset: dashOffset }}
              transition={{ duration: 1, ease: 'easeOut' }}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center font-mono text-xl font-bold text-text-primary">
            {Math.round(clampedValue)}%
          </div>
        </div>
        {label && (
          <div className="mt-2 text-sm text-text-secondary text-center font-body">
            {label}
          </div>
        )}
      </div>
    );
  }
);

ConfidenceGauge.displayName = 'ConfidenceGauge';
