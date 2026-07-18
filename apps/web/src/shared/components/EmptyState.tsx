import type { HTMLAttributes } from 'react';
import { forwardRef } from 'react';
import type { LucideIcon } from 'lucide-react';
import { Inbox } from 'lucide-react';

export interface EmptyStateProps extends HTMLAttributes<HTMLDivElement> {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export const EmptyState = forwardRef<HTMLDivElement, EmptyStateProps>(
  ({ icon: Icon = Inbox, title, description, action, className = '', ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`flex flex-col items-center justify-center gap-4 py-12 px-6 text-center ${className}`}
        {...props}
      >
        <Icon
          size={48}
          strokeWidth={1}
          className="text-[var(--text-secondary)] opacity-60"
        />
        <div className="flex flex-col gap-1.5">
          <h3 className="text-base font-medium text-[var(--text-primary)]">
            {title}
          </h3>
          {description && (
            <p className="text-sm text-[var(--text-secondary)] max-w-[280px]">
              {description}
            </p>
          )}
        </div>
        {action && (
          <button
            onClick={action.onClick}
            className="mt-2 px-4 py-2 rounded-xl text-sm font-medium
                       text-[var(--accent-primary)] border border-[var(--accent-primary)]
                       bg-[rgba(0,229,255,0.06)] hover:bg-[rgba(0,229,255,0.12)]
                       transition-colors"
          >
            {action.label}
          </button>
        )}
      </div>
    );
  }
);

EmptyState.displayName = 'EmptyState';
