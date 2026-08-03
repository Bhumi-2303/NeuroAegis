import type { HTMLAttributes } from 'react';
import { forwardRef } from 'react';
import { AlertTriangle } from 'lucide-react';

export interface ErrorStateProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  message?: string;
  onRetry?: () => void;
  retryLabel?: string;
}

export const ErrorState = forwardRef<HTMLDivElement, ErrorStateProps>(
  (
    {
      title = 'Something went wrong',
      message,
      onRetry,
      retryLabel = 'Retry',
      className = '',
      ...props
    },
    ref
  ) => {
    return (
      <div
        ref={ref}
        className={`flex flex-col items-center justify-center gap-4 py-12 px-6 text-center
                     border border-[var(--bg-3)] bg-[var(--bg-1)] rounded-2xl ${className}`}
        {...props}
      >
        <AlertTriangle
          size={48}
          strokeWidth={1}
          className="text-[var(--text-secondary)] opacity-80"
        />
        <div className="flex flex-col gap-1.5">
          <h3 className="text-base font-medium text-[var(--text-primary)]">
            {title}
          </h3>
          {message && (
            <p className="text-sm text-[var(--text-secondary)] max-w-[320px]">
              {message}
            </p>
          )}
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="mt-2 px-4 py-2 rounded-xl text-sm font-medium
                       text-[var(--text-primary)] border border-[var(--bg-3)]
                       bg-[var(--bg-2)] hover:bg-[var(--bg-3)]
                       transition-colors"
          >
            {retryLabel}
          </button>
        )}
      </div>
    );
  }
);

ErrorState.displayName = 'ErrorState';
