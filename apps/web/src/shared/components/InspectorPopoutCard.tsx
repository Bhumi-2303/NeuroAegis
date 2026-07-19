import type { HTMLAttributes, ReactNode } from 'react';
import { forwardRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface InspectorPopoutCardProps extends HTMLAttributes<HTMLDivElement> {
  visible: boolean;
  children: ReactNode;
  liveDot?: boolean;
  className?: string;
}

/**
 * InspectorPopoutCard — floating card that visually breaks out
 * past its parent panel's edge (overlap, elevated shadow, glowing border,
 * optional pulsing "Live" dot).
 */
export const InspectorPopoutCard = forwardRef<HTMLDivElement, InspectorPopoutCardProps>(
  ({ visible, children, liveDot = false, className = '', ...props }, ref) => {
    return (
      <AnimatePresence>
        {visible && (
          <motion.div
            ref={ref}
            initial={{ opacity: 0, y: 12, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.97 }}
            transition={{ type: 'spring', stiffness: 400, damping: 28 }}
            className={`absolute z-50 ${className}`}
            style={{
              background: 'rgba(7, 17, 29, 0.92)',
              backdropFilter: 'blur(24px)',
              WebkitBackdropFilter: 'blur(24px)',
              border: '1px solid rgba(0, 229, 255, 0.25)',
              borderRadius: 16,
              boxShadow: [
                '0 0 30px rgba(0, 229, 255, 0.12)',
                '0 20px 60px rgba(0, 0, 0, 0.5)',
                'inset 0 1px 0 rgba(255, 255, 255, 0.06)',
              ].join(', '),
              padding: '16px',
            }}
            {...(props as any)}
          >
            {/* Live dot */}
            {liveDot && (
              <div className="absolute top-3 right-3 flex items-center gap-1.5">
                <span className="relative flex h-2 w-2">
                  <span
                    className="absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping"
                    style={{ backgroundColor: 'var(--state-success)' }}
                  />
                  <span
                    className="relative inline-flex h-2 w-2 rounded-full"
                    style={{
                      backgroundColor: 'var(--state-success)',
                      boxShadow: '0 0 6px var(--state-success)',
                    }}
                  />
                </span>
                <span className="text-[10px] font-mono uppercase tracking-wider text-[var(--state-success)]">
                  Live
                </span>
              </div>
            )}

            {children}
          </motion.div>
        )}
      </AnimatePresence>
    );
  }
);

InspectorPopoutCard.displayName = 'InspectorPopoutCard';
