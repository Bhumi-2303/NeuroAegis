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
      className={`bg-white rounded-2xl border border-gray-200/60 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.05)] overflow-hidden flex flex-col ${className}`}
    >
      {(title || headerRight) && (
        <div className="flex items-center justify-between px-6 py-5 border-b border-gray-50/50 bg-gray-50/30">
          <div className="flex items-center gap-2">
            {icon && <span className="text-gray-400">{icon}</span>}
            <h3 className="text-[13px] font-semibold text-gray-700 tracking-wide uppercase">{title}</h3>
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
