import type { HTMLAttributes } from 'react';
import { forwardRef } from 'react';
import { motion } from 'framer-motion';

export interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  icon?: React.ReactNode;
  interactive?: boolean;
  layoutId?: string;
}

export const GlassCard = forwardRef<HTMLDivElement, GlassCardProps>(
  ({ children, title, icon, interactive = false, layoutId, className = '', ...props }, ref) => {
    return (
      <motion.div
        ref={ref}
        layoutId={layoutId}
        whileHover={interactive ? { scale: 1.02, y: -2 } : {}}
        whileTap={interactive ? { scale: 0.98 } : {}}
        transition={{ type: 'spring', stiffness: 400, damping: 25 }}
        className={`glass-card p-4 flex flex-col gap-4 relative overflow-hidden ${interactive ? 'cursor-pointer hover:shadow-[0_0_30px_rgba(0,229,255,0.15)]' : ''} ${className}`}
        {...(props as any)}
      >
        {(title || icon) && (
          <div className="flex items-center gap-2 border-b border-[rgba(255,255,255,0.08)] pb-3 mb-1">
            {icon && <span className="text-[var(--text-secondary)]">{icon}</span>}
            {title && <h3 className="text-[var(--text-primary)] font-semibold text-sm m-0">{title}</h3>}
          </div>
        )}
        <div className="flex-1 flex flex-col relative z-10">
          {children}
        </div>
      </motion.div>
    );
  }
);

GlassCard.displayName = 'GlassCard';
