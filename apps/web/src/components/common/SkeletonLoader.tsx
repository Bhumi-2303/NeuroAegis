import React from 'react';

export const SkeletonLoader: React.FC = () => {
  return (
    <div className="p-4 lg:p-6 max-w-7xl mx-auto space-y-6 animate-pulse" aria-label="Loading clinical interface">
      {/* Top Banner Skeleton */}
      <div className="h-16 bg-slate-900/90 rounded-xl border border-slate-800 w-full" />

      {/* Grid skeleton */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Main Waveform Screen (8 cols) */}
        <div className="lg:col-span-8 bg-slate-900/80 rounded-xl border border-slate-800 p-5 space-y-4">
          <div className="flex justify-between items-center">
            <div className="h-5 bg-slate-800 rounded w-52" />
            <div className="h-5 bg-slate-800 rounded w-28" />
          </div>
          {/* 10 Channel wave bars */}
          <div className="space-y-3 pt-2">
            {Array.from({ length: 10 }).map((_, i) => (
              <div key={`skel-chan-${i}`} className="h-8 bg-slate-950/60 rounded flex items-center px-4">
                <div className="h-3 bg-slate-800 rounded w-14" />
                <div className="ml-4 flex-1 h-0.5 bg-slate-800/40" />
              </div>
            ))}
          </div>
        </div>

        {/* Right Analytics Sidebar (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-slate-900/80 rounded-xl border border-slate-800 p-5 space-y-4">
            <div className="h-4 bg-slate-800 rounded w-36" />
            <div className="h-32 bg-slate-800/40 rounded-full mx-auto w-32" />
            <div className="h-4 bg-slate-800 rounded w-full" />
          </div>
          <div className="bg-slate-900/80 rounded-xl border border-slate-800 p-5 space-y-3">
            <div className="h-4 bg-slate-800 rounded w-44" />
            <div className="h-4 bg-slate-800/60 rounded w-full" />
            <div className="h-4 bg-slate-800/60 rounded w-5/6" />
            <div className="h-4 bg-slate-800/60 rounded w-4/6" />
          </div>
        </div>
      </div>
    </div>
  );
};
