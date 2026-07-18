import { forwardRef } from 'react';
import type { ReactNode, HTMLAttributes } from 'react';
import { motion } from 'framer-motion';

export interface NeonBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'highlight' | 'success' | 'warning' | 'danger';
  className?: string;
  interactive?: boolean;
}

const variantColors = {
  primary: 'var(--accent-primary)',
  secondary: 'var(--accent-secondary)',
  highlight: 'var(--accent-highlight)',
  success: 'var(--state-success)',
  warning: 'var(--state-warning)',
  danger: 'var(--state-danger)'
};

export const NeonBadge = forwardRef<HTMLSpanElement, NeonBadgeProps>(
  ({ children, variant = 'primary', className = '', interactive = false, ...props }, ref) => {
    const color = variantColors[variant];

    return (
      <motion.span
        ref={ref}
        whileHover={interactive ? { scale: 1.05 } : {}}
        whileTap={interactive ? { scale: 0.95 } : {}}
        className={`inline-flex items-center justify-center rounded-full px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide ${interactive ? 'cursor-pointer' : ''} ${className}`}
        style={{
          color,
          backgroundColor: `color-mix(in srgb, ${color} 15%, transparent)`,
          borderColor: `color-mix(in srgb, ${color} 30%, transparent)`,
          borderWidth: '1px',
          textShadow: `0 0 8px color-mix(in srgb, ${color} 50%, transparent)`
        }}
        {...(props as any)}
      >
        {children}
      </motion.span>
    );
  }
);

NeonBadge.displayName = 'NeonBadge';
