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

export const ShapBarChart = forwardRef<HTMLDivElement, ShapBarChartProps>(
  ({ features, maxBars = 8, className = '', ...props }, ref) => {
    const sortedFeatures = [...features]
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, maxBars);

    const maxAbs = Math.max(...sortedFeatures.map((f) => Math.abs(f.value)), 1);

    const containerVariants = {
      hidden: { opacity: 0 },
      show: {
        opacity: 1,
        transition: {
          staggerChildren: 0.05
        }
      }
    };

    const itemVariants = {
      hidden: { opacity: 0, scaleX: 0 },
      show: { opacity: 1, scaleX: 1 }
    };

    return (
      <div ref={ref} className={`flex flex-col gap-2 ${className}`} {...props}>
        <motion.div
          className="flex flex-col gap-2 w-full"
          variants={containerVariants}
          initial="hidden"
          animate="show"
        >
        {sortedFeatures.map((feature, i) => {
          const isPositive = feature.value >= 0;
          const widthPercent = (Math.abs(feature.value) / maxAbs) * 100;
          const color = isPositive ? 'var(--accent-primary)' : 'var(--state-danger)';

          return (
            <div key={`${feature.name}-${i}`} className="flex items-center gap-4 text-xs font-mono">
              <div className="w-24 shrink-0 truncate text-right text-text-secondary">
                {feature.name}
              </div>
              <div className="relative flex-1 h-3 flex items-center">
                {/* Center line */}
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-text-secondary/20" />
                
                <div className="w-1/2 flex justify-end pr-1">
                  {!isPositive && (
                    <motion.div
                      variants={itemVariants}
                      className="h-full rounded-l-full"
                      style={{
                        width: `${widthPercent}%`,
                        backgroundColor: color,
                        boxShadow: `0 0 6px color-mix(in srgb, ${color} 50%, transparent)`,
                        transformOrigin: 'right'
                      }}
                    />
                  )}
                </div>
                
                <div className="w-1/2 flex justify-start pl-1">
                  {isPositive && (
                    <motion.div
                      variants={itemVariants}
                      className="h-full rounded-r-full"
                      style={{
                        width: `${widthPercent}%`,
                        backgroundColor: color,
                        boxShadow: `0 0 6px color-mix(in srgb, ${color} 50%, transparent)`,
                        transformOrigin: 'left'
                      }}
                    />
                  )}
                </div>
              </div>
            </div>
          );
        })}
        </motion.div>
      </div>
    );
  }
);

ShapBarChart.displayName = 'ShapBarChart';
