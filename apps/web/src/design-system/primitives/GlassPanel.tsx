import type { HTMLAttributes } from 'react';
import { forwardRef } from 'react';
import { motion } from 'framer-motion';

export interface GlassPanelProps extends HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  interactive?: boolean;
}

export const GlassPanel = forwardRef<HTMLDivElement, GlassPanelProps>(
  ({ children, interactive = false, className = '', ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`glass-card relative overflow-hidden ${interactive ? 'cursor-pointer hover:shadow-[0_0_24px_rgba(20,184,166,0.12)] hover:-translate-y-[2px]' : ''} ${className}`}
        {...props}
      >
        {children}
      </div>
    );
  }
);

GlassPanel.displayName = 'GlassPanel';

export const MotionGlassPanel = motion.create(GlassPanel);
