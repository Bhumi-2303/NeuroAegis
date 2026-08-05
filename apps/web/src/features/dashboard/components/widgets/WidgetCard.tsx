import React from 'react';
import { motion } from 'framer-motion';

interface WidgetCardProps {
  title?: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  headerRight?: React.ReactNode;
}

export function WidgetCard({ title, icon, children, className = '', headerRight }: WidgetCardProps) {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={`glass-card overflow-hidden flex flex-col ${className}`}
    >
      {(title || headerRight) && (
        <div className="flex items-center justify-between px-6 py-5 border-b border-[var(--bg-4)] bg-[var(--bg-3)]">
          <div className="flex items-center gap-2">
            {icon && <span className="text-[var(--text-secondary)]">{icon}</span>}
            <h3 className="text-[13px] font-semibold text-[var(--text-primary)] tracking-wide uppercase">{title}</h3>
          </div>
          {headerRight && <div>{headerRight}</div>}
        </div>
      )}
      <div className="p-6 flex-1 flex flex-col">
        {children}
      </div>
    </motion.div>
  );
}
