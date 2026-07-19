import { forwardRef } from 'react';
import type { HTMLAttributes } from 'react';
import { motion } from 'framer-motion';

export interface ConfidenceGaugeProps extends HTMLAttributes<HTMLDivElement> {
  value: number;
  size?: number;
  label?: string;
  className?: string;
}

/**
 * CircularScoreGauge — ring + big centered number, paired with a
 * vertical gradient scale bar (teal→yellow→orange→red) beside it.
 * 
 * Preserves the original ConfidenceGauge prop interface.
 */
export const ConfidenceGauge = forwardRef<HTMLDivElement, ConfidenceGaugeProps>(
  ({ value, size = 120, label, className = '', ...props }, ref) => {
    const clampedValue = Math.max(0, Math.min(100, value));

    // Dynamic color based on value
    const getColor = (v: number): string => {
      if (v >= 80) return '#00E5FF';     // teal
      if (v >= 60) return '#00FFA3';     // green
      if (v >= 40) return '#FFB020';     // yellow/orange
      return '#FF4D6D';                   // red
    };

    const color = getColor(clampedValue);

    const strokeWidth = size * 0.06;
    const radius = (size - strokeWidth * 2) / 2;
    const center = size / 2;

    // Full 360 ring
    const circumference = 2 * Math.PI * radius;
    const dashOffset = circumference - (clampedValue / 100) * circumference;

    // Scale bar dimensions
    const scaleBarHeight = size * 0.75;
    const scaleBarWidth = 6;

    // Position marker on scale bar
    const markerY = scaleBarHeight - (clampedValue / 100) * scaleBarHeight;

    return (
      <div ref={ref} className={`flex items-center gap-4 ${className}`} {...props}>
        {/* Ring gauge */}
        <div className="relative" style={{ width: size, height: size }}>
          {/* Outer glow ring */}
          <div
            className="absolute inset-0 rounded-full"
            style={{
              boxShadow: `0 0 ${size * 0.15}px color-mix(in srgb, ${color} 25%, transparent)`,
            }}
          />

          <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
            {/* Background track */}
            <circle
              cx={center}
              cy={center}
              r={radius}
              fill="none"
              stroke="rgba(255,255,255,0.04)"
              strokeWidth={strokeWidth}
            />

            {/* Animated ring */}
            <motion.circle
              cx={center}
              cy={center}
              r={radius}
              fill="none"
              stroke={color}
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              strokeLinecap="round"
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset: dashOffset }}
              transition={{ duration: 1.2, ease: 'easeOut' }}
              style={{
                filter: `drop-shadow(0 0 6px ${color})`,
              }}
            />
          </svg>

          {/* Centered number */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.span
              className="font-mono font-bold text-[var(--text-primary)]"
              style={{ fontSize: size * 0.28, lineHeight: 1 }}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3, duration: 0.5 }}
            >
              {Math.round(clampedValue)}
            </motion.span>
            <span
              className="font-mono text-[var(--text-secondary)]"
              style={{ fontSize: size * 0.1 }}
            >
              %
            </span>
          </div>
        </div>

        {/* Vertical gradient scale bar */}
        <div className="flex flex-col items-center gap-1" style={{ height: scaleBarHeight }}>
          {/* Scale labels */}
          <span className="text-[8px] font-mono text-[var(--text-secondary)]">100</span>

          <div className="relative flex-1 flex flex-col items-center">
            {/* Gradient bar */}
            <div
              className="rounded-full overflow-hidden"
              style={{
                width: scaleBarWidth,
                height: '100%',
                background: 'linear-gradient(to bottom, #00E5FF, #00FFA3, #FFB020, #FF4D6D)',
                opacity: 0.6,
              }}
            />

            {/* Position marker */}
            <motion.div
              className="absolute -left-1"
              style={{ width: scaleBarWidth + 8 }}
              initial={{ top: '100%' }}
              animate={{ top: `${markerY / scaleBarHeight * 100}%` }}
              transition={{ duration: 1, ease: 'easeOut' }}
            >
              <div
                className="w-full h-1 rounded-full"
                style={{
                  backgroundColor: color,
                  boxShadow: `0 0 6px ${color}`,
                }}
              />
            </motion.div>
          </div>

          <span className="text-[8px] font-mono text-[var(--text-secondary)]">0</span>
        </div>

        {/* Label */}
        {label && (
          <div className="flex flex-col gap-0.5 ml-1">
            <span
              className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-secondary)]"
            >
              Status
            </span>
            <span
              className="text-xs font-semibold"
              style={{ color }}
            >
              {label}
            </span>
          </div>
        )}
      </div>
    );
  }
);

ConfidenceGauge.displayName = 'ConfidenceGauge';
