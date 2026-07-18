import type { HTMLAttributes } from 'react';
import { forwardRef } from 'react';

export interface NeonBorderProps extends HTMLAttributes<HTMLDivElement> {
  color?: 'primary' | 'secondary' | 'highlight' | 'success' | 'warning' | 'danger';
}

const colorMap = {
  primary: 'rgba(0, 229, 255, 0.5)',
  secondary: 'rgba(75, 125, 255, 0.5)',
  highlight: 'rgba(139, 92, 246, 0.5)',
  success: 'rgba(0, 255, 163, 0.5)',
  warning: 'rgba(255, 176, 32, 0.5)',
  danger: 'rgba(255, 77, 109, 0.5)',
};

export const NeonBorder = forwardRef<HTMLDivElement, NeonBorderProps>(
  ({ children, color = 'primary', className = '', ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`relative p-[1px] rounded-2xl overflow-hidden group ${className}`}
        {...props}
      >
        <div 
          className="absolute inset-0 z-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
          style={{
            background: `conic-gradient(from 0deg, transparent 0 340deg, ${colorMap[color]} 360deg)`,
            animation: 'spin 3s linear infinite',
          }}
        />
        <div className="relative z-10 w-full h-full bg-[var(--bg-3)] rounded-[15px] overflow-hidden">
          {children}
        </div>
      </div>
    );
  }
);

NeonBorder.displayName = 'NeonBorder';
