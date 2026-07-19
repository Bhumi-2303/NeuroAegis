import type { HTMLAttributes, ReactNode } from 'react';
import { forwardRef } from 'react';

export interface HudCornerFrameProps extends HTMLAttributes<HTMLDivElement> {
  label?: string;
  icon?: ReactNode;
  children?: ReactNode;
  className?: string;
}

/**
 * HudCornerFrame — thin corner-bracket frame around section titles.
 * Provides a futuristic HUD-style framing element for panel headers.
 */
export const HudCornerFrame = forwardRef<HTMLDivElement, HudCornerFrameProps>(
  ({ label, icon, children, className = '', ...props }, ref) => {
    const cornerSize = 12;
    const borderColor = 'rgba(0, 229, 255, 0.35)';
    const glowColor = 'rgba(0, 229, 255, 0.15)';

    const cornerStyle = (
      top: boolean,
      left: boolean
    ): React.CSSProperties => ({
      position: 'absolute',
      width: cornerSize,
      height: cornerSize,
      ...(top ? { top: 0 } : { bottom: 0 }),
      ...(left ? { left: 0 } : { right: 0 }),
      borderColor: borderColor,
      borderStyle: 'solid',
      borderWidth: 0,
      ...(top && left
        ? { borderTopWidth: 2, borderLeftWidth: 2 }
        : top && !left
          ? { borderTopWidth: 2, borderRightWidth: 2 }
          : !top && left
            ? { borderBottomWidth: 2, borderLeftWidth: 2 }
            : { borderBottomWidth: 2, borderRightWidth: 2 }),
      filter: `drop-shadow(0 0 3px ${glowColor})`,
    });

    return (
      <div
        ref={ref}
        className={`relative ${className}`}
        {...props}
      >
        {/* Corner brackets */}
        <div style={cornerStyle(true, true)} aria-hidden="true" />
        <div style={cornerStyle(true, false)} aria-hidden="true" />
        <div style={cornerStyle(false, true)} aria-hidden="true" />
        <div style={cornerStyle(false, false)} aria-hidden="true" />

        {/* Header row */}
        {(label || icon) && (
          <div className="flex items-center gap-2 px-4 pt-3 pb-2">
            {icon && (
              <span className="text-[var(--accent-primary)] opacity-70">
                {icon}
              </span>
            )}
            {label && (
              <span
                className="text-[10px] font-mono uppercase tracking-[0.2em] text-[var(--accent-primary)]"
                style={{ textShadow: '0 0 8px rgba(0, 229, 255, 0.4)' }}
              >
                {label}
              </span>
            )}
          </div>
        )}

        {/* Content */}
        <div className="px-4 pb-3">
          {children}
        </div>
      </div>
    );
  }
);

HudCornerFrame.displayName = 'HudCornerFrame';
