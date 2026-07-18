import type { HTMLAttributes } from 'react';
import { forwardRef } from 'react';

export interface SkeletonShimmerProps extends HTMLAttributes<HTMLDivElement> {}

export const SkeletonShimmer = forwardRef<HTMLDivElement, SkeletonShimmerProps>(
  ({ className = '', ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`relative overflow-hidden bg-[rgba(255,255,255,0.02)] rounded-md ${className}`}
        {...props}
      >
        <div 
          className="absolute inset-0 -translate-x-full"
          style={{
            backgroundImage: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent)',
            animation: 'shimmer 1.5s infinite linear'
          }}
        />
      </div>
    );
  }
);

SkeletonShimmer.displayName = 'SkeletonShimmer';
