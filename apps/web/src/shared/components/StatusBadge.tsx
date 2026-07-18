import { forwardRef } from 'react';
import type { HTMLAttributes } from 'react';
import { motion } from 'framer-motion';

export interface StatusBadgeProps extends HTMLAttributes<HTMLDivElement> {
  status: 'normal' | 'warning' | 'critical' | 'offline';
  label?: string;
  pulse?: boolean;
  className?: string;
}

const statusColors = {
  normal: 'var(--state-success)',
  warning: 'var(--state-warning)',
  critical: 'var(--state-danger)',
  offline: 'var(--text-secondary)'
};

export const StatusBadge = forwardRef<HTMLDivElement, StatusBadgeProps>(
  ({ status, label, pulse, className = '', ...props }, ref) => {
    const color = statusColors[status];
    const shouldPulse = pulse ?? status === 'critical';

    return (
      <div ref={ref} className={`inline-flex items-center gap-2 ${className}`} {...props}>
        <div className="relative flex items-center justify-center h-2 w-2">
          {shouldPulse && (
            <motion.div
              className="absolute h-full w-full rounded-full"
              style={{ backgroundColor: color }}
              animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            />
          )}
          <div
            className="relative h-2 w-2 rounded-full"
            style={{ backgroundColor: color }}
          />
        </div>
        {label && <span className="text-sm text-text-secondary">{label}</span>}
      </div>
    );
  }
);

StatusBadge.displayName = 'StatusBadge';
