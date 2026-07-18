import { forwardRef } from 'react';
import type { ReactNode, HTMLAttributes } from 'react';

export interface NeonBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'highlight' | 'success' | 'warning' | 'danger';
  className?: string;
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
  ({ children, variant = 'primary', className = '', ...props }, ref) => {
    const color = variantColors[variant];

    return (
      <span
        ref={ref}
        className={`inline-flex items-center justify-center rounded-full px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide ${className}`}
        style={{
          color,
          backgroundColor: `color-mix(in srgb, ${color} 15%, transparent)`,
          borderColor: `color-mix(in srgb, ${color} 30%, transparent)`,
          borderWidth: '1px',
          textShadow: `0 0 8px color-mix(in srgb, ${color} 50%, transparent)`
        }}
        {...props}
      >
        {children}
      </span>
    );
  }
);

NeonBadge.displayName = 'NeonBadge';
