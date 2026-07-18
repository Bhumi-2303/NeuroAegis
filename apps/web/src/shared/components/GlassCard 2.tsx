import type { HTMLAttributes } from 'react';
import { forwardRef } from 'react';
import { GlassPanel } from '../../design-system/primitives';

export interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  icon?: React.ReactNode;
}

export const GlassCard = forwardRef<HTMLDivElement, GlassCardProps>(
  ({ children, title, icon, className = '', ...props }, ref) => {
    return (
      <GlassPanel ref={ref} className={`p-4 flex flex-col gap-4 ${className}`} {...props}>
        {(title || icon) && (
          <div className="flex items-center gap-2 border-b border-[rgba(255,255,255,0.08)] pb-3 mb-1">
            {icon && <span className="text-[var(--text-secondary)]">{icon}</span>}
            {title && <h3 className="text-[var(--text-primary)] font-semibold text-sm m-0">{title}</h3>}
          </div>
        )}
        <div className="flex-1 flex flex-col relative z-10">
          {children}
        </div>
      </GlassPanel>
    );
  }
);

GlassCard.displayName = 'GlassCard';
