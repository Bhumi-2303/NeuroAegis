import type { HTMLAttributes } from 'react';
import { forwardRef } from 'react';
import { motion } from 'framer-motion';

export interface SessionThumbnail {
  id: string;
  label: string;
  timestamp?: string;
}

export interface SessionFilmstripProps extends HTMLAttributes<HTMLDivElement> {
  sessions: SessionThumbnail[];
  activeId?: string;
  onSelect?: (id: string) => void;
  className?: string;
}

/**
 * SessionFilmstrip — horizontal scrollable thumbnail strip of past
 * sessions with a count label above it, active one highlighted.
 */
export const SessionFilmstrip = forwardRef<HTMLDivElement, SessionFilmstripProps>(
  ({ sessions, activeId, onSelect, className = '', ...props }, ref) => {
    if (!sessions || sessions.length === 0) return null;

    return (
      <div ref={ref} className={`flex flex-col gap-2 ${className}`} {...props}>
        {/* Count label */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono uppercase tracking-[0.15em] text-[var(--text-secondary)]">
            Sessions
          </span>
          <span
            className="text-[10px] font-mono px-1.5 py-0.5 rounded-full"
            style={{
              backgroundColor: 'rgba(0, 229, 255, 0.1)',
              color: 'var(--accent-primary)',
              border: '1px solid rgba(0, 229, 255, 0.2)',
            }}
          >
            {sessions.length}
          </span>
        </div>

        {/* Scrollable strip */}
        <div
          className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin"
          style={{
            scrollbarWidth: 'thin',
            scrollbarColor: 'rgba(0, 229, 255, 0.2) transparent',
          }}
        >
          {sessions.map((session) => {
            const isActive = session.id === activeId;
            return (
              <motion.button
                key={session.id}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => onSelect?.(session.id)}
                className={`relative flex flex-col items-center justify-center shrink-0 w-20 h-14 rounded-lg transition-all duration-200 outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-primary)] ${
                  isActive
                    ? 'border-[var(--accent-primary)]'
                    : 'border-[rgba(255,255,255,0.06)] hover:border-[rgba(255,255,255,0.15)]'
                }`}
                style={{
                  background: isActive
                    ? 'rgba(0, 229, 255, 0.08)'
                    : 'rgba(11, 22, 37, 0.6)',
                  border: `1px solid ${
                    isActive ? 'rgba(0, 229, 255, 0.4)' : 'rgba(255, 255, 255, 0.06)'
                  }`,
                  boxShadow: isActive ? '0 0 12px rgba(0, 229, 255, 0.15)' : 'none',
                }}
              >
                <span
                  className="text-[10px] font-mono truncate max-w-[68px]"
                  style={{
                    color: isActive ? 'var(--accent-primary)' : 'var(--text-secondary)',
                  }}
                >
                  {session.label}
                </span>
                {session.timestamp && (
                  <span className="text-[8px] font-mono text-[var(--text-secondary)] opacity-60 mt-0.5">
                    {session.timestamp}
                  </span>
                )}
                {isActive && (
                  <motion.div
                    layoutId="filmstrip-active"
                    className="absolute -bottom-0.5 left-1/2 -translate-x-1/2 w-4 h-0.5 rounded-full"
                    style={{
                      background: 'var(--accent-primary)',
                      boxShadow: '0 0 6px var(--accent-primary)',
                    }}
                  />
                )}
              </motion.button>
            );
          })}
        </div>
      </div>
    );
  }
);

SessionFilmstrip.displayName = 'SessionFilmstrip';
