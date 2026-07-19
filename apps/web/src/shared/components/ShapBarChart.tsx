import { forwardRef } from 'react';
import type { HTMLAttributes } from 'react';
import { motion } from 'framer-motion';

export interface ShapFeature {
  name: string;
  value: number;
}

export interface ShapBarChartProps extends HTMLAttributes<HTMLDivElement> {
  features: ShapFeature[];
  maxBars?: number;
  className?: string;
}

/**
 * ShapFeatureRow — repeating list row: label + horizontal track that
 * shades teal→yellow→orange→red + a marker positioned by the signed
 * SHAP value.
 * 
 * Preserves the original ShapBarChart prop interface (features, maxBars).
 */
export const ShapBarChart = forwardRef<HTMLDivElement, ShapBarChartProps>(
  ({ features, maxBars = 8, className = '', ...props }, ref) => {
    const sortedFeatures = [...features]
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, maxBars);

    const maxAbs = Math.max(...sortedFeatures.map((f) => Math.abs(f.value)), 0.001);

    const containerVariants = {
      hidden: { opacity: 0 },
      show: {
        opacity: 1,
        transition: {
          staggerChildren: 0.06,
        },
      },
    };

    const itemVariants = {
      hidden: { opacity: 0, x: -8 },
      show: { opacity: 1, x: 0 },
    };

    /**
     * Map a normalized [0..1] absolute value to a color on the
     * teal→green→yellow→orange→red scale.
     */
    const getTrackColor = (normalizedAbs: number): string => {
      if (normalizedAbs < 0.25) return '#00E5FF';
      if (normalizedAbs < 0.5) return '#00FFA3';
      if (normalizedAbs < 0.75) return '#FFB020';
      return '#FF4D6D';
    };

    return (
      <div ref={ref} className={`flex flex-col gap-1 ${className}`} {...props}>
        <motion.div
          className="flex flex-col gap-0.5 w-full"
          variants={containerVariants}
          initial="hidden"
          animate="show"
        >
          {sortedFeatures.map((feature, i) => {
            const normalizedAbs = Math.abs(feature.value) / maxAbs;
            const isPositive = feature.value >= 0;
            const markerColor = getTrackColor(normalizedAbs);

            // Position marker: 50% is center, positive goes right, negative goes left
            const markerPercent = 50 + (feature.value / maxAbs) * 50;

            return (
              <motion.div
                key={`${feature.name}-${i}`}
                variants={itemVariants}
                className="flex items-center gap-3 py-1.5 px-2 rounded-lg hover:bg-[rgba(255,255,255,0.02)] transition-colors group"
              >
                {/* Feature label */}
                <div className="w-28 shrink-0 truncate text-right text-[11px] font-mono text-[var(--text-secondary)] group-hover:text-[var(--text-primary)] transition-colors">
                  {feature.name}
                </div>

                {/* Track */}
                <div className="relative flex-1 h-5 rounded-full overflow-hidden">
                  {/* Background track with gradient */}
                  <div
                    className="absolute inset-0 rounded-full"
                    style={{
                      background:
                        'linear-gradient(90deg, #FF4D6D15, #FFB02010, #00FFA308, #00E5FF05, #00FFA308, #FFB02010, #FF4D6D15)',
                    }}
                  />

                  {/* Center line */}
                  <div className="absolute left-1/2 top-1 bottom-1 w-px bg-[rgba(255,255,255,0.12)]" />

                  {/* Active fill from center */}
                  <motion.div
                    className="absolute top-0.5 bottom-0.5 rounded-full"
                    initial={{
                      left: '50%',
                      width: '0%',
                    }}
                    animate={{
                      left: isPositive ? '50%' : `${markerPercent}%`,
                      width: `${normalizedAbs * 50}%`,
                    }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                    style={{
                      background: `linear-gradient(${isPositive ? '90deg' : '270deg'}, ${markerColor}40, ${markerColor}20)`,
                    }}
                  />

                  {/* Marker dot */}
                  <motion.div
                    className="absolute top-1/2 -translate-y-1/2"
                    initial={{ left: '50%' }}
                    animate={{ left: `${markerPercent}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                  >
                    <div
                      className="w-2.5 h-2.5 rounded-full -translate-x-1/2"
                      style={{
                        backgroundColor: markerColor,
                        boxShadow: `0 0 8px ${markerColor}`,
                      }}
                    />
                  </motion.div>
                </div>

                {/* Value readout */}
                <div className="w-14 shrink-0 text-right">
                  <span
                    className="text-[11px] font-mono font-semibold"
                    style={{ color: markerColor }}
                  >
                    {feature.value >= 0 ? '+' : ''}
                    {feature.value.toFixed(3)}
                  </span>
                </div>
              </motion.div>
            );
          })}
        </motion.div>
      </div>
    );
  }
);

ShapBarChart.displayName = 'ShapBarChart';
