import type { HTMLAttributes } from 'react';
import { forwardRef, useId } from 'react';
import { motion } from 'framer-motion';

export interface FrequencyBandRowProps extends HTMLAttributes<HTMLDivElement> {
  bandName: string;
  greekLetter: string;
  color: string;
  hzRange: string;
  sparklineData: number[];
  avgPower: number;
  className?: string;
}

/**
 * FrequencyBandRow — glowing sparkline + Hz readout + colored
 * Greek-letter icon chip (γ β α δ θ), one per band.
 */
export const FrequencyBandRow = forwardRef<HTMLDivElement, FrequencyBandRowProps>(
  (
    {
      bandName,
      greekLetter,
      color,
      hzRange,
      sparklineData,
      avgPower,
      className = '',
      ...props
    },
    ref
  ) => {
    const gradId = useId();

    // Generate sparkline path
    const width = 140;
    const height = 28;
    let pathD = '';

    if (sparklineData.length > 1) {
      const min = Math.min(...sparklineData);
      const max = Math.max(...sparklineData);
      const range = max - min || 1;

      const points = sparklineData.map((val, i) => {
        const x = (i / (sparklineData.length - 1)) * width;
        const y = height - ((val - min) / range) * (height - 4) - 2;
        return { x, y };
      });

      pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
    }

    return (
      <motion.div
        ref={ref}
        initial={{ opacity: 0, x: -12 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
        className={`flex items-center gap-3 py-2 px-3 rounded-xl hover:bg-[rgba(255,255,255,0.02)] transition-colors ${className}`}
        {...(props as any)}
      >
        {/* Greek letter chip */}
        <div
          className="flex items-center justify-center w-8 h-8 rounded-lg text-sm font-bold shrink-0"
          style={{
            backgroundColor: `color-mix(in srgb, ${color} 15%, transparent)`,
            color: color,
            border: `1px solid color-mix(in srgb, ${color} 25%, transparent)`,
            textShadow: `0 0 8px ${color}`,
          }}
        >
          {greekLetter}
        </div>

        {/* Band name + Hz range */}
        <div className="flex flex-col min-w-[72px]">
          <span className="text-xs font-semibold text-[var(--text-primary)]">
            {bandName}
          </span>
          <span className="text-[10px] font-mono text-[var(--text-secondary)]">
            {hzRange}
          </span>
        </div>

        {/* Sparkline */}
        <div className="flex-1">
          <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
            <defs>
              <linearGradient id={gradId} x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor={color} stopOpacity={0.1} />
                <stop offset="100%" stopColor={color} stopOpacity={0.5} />
              </linearGradient>
            </defs>
            {pathD && (
              <>
                <path
                  d={`${pathD} L ${width} ${height} L 0 ${height} Z`}
                  fill={`url(#${gradId})`}
                />
                <path
                  d={pathD}
                  fill="none"
                  stroke={color}
                  strokeWidth={1.5}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  style={{ filter: `drop-shadow(0 0 4px ${color})` }}
                />
              </>
            )}
          </svg>
        </div>

        {/* Power readout */}
        <div className="text-right min-w-[64px] shrink-0">
          <span className="text-sm font-mono font-semibold" style={{ color }}>
            {avgPower.toFixed(2)}
          </span>
          <span className="text-[10px] font-mono text-[var(--text-secondary)] ml-0.5">µV²</span>
        </div>
      </motion.div>
    );
  }
);

FrequencyBandRow.displayName = 'FrequencyBandRow';
