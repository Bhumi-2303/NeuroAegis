import { useState } from 'react';
import { motion } from 'framer-motion';
import { Brain, RotateCw, ZoomIn, Crosshair } from 'lucide-react';
import { GlassCard, Scene } from '../../../shared/components';
import { slideUp } from '../../../shared/lib/motion-presets';

export function BrainAnalysisPage(): React.JSX.Element {
  const [autoRotate, setAutoRotate] = useState(true);
  const [highlightRegion, setHighlightRegion] = useState(false);
  const [zoomKey, setZoomKey] = useState(0);

  return (
    <motion.div
      className="flex flex-col gap-6 p-6"
      {...slideUp}
    >
      {/* Page Header */}
      <div className="flex items-center gap-3">
        <Brain size={28} strokeWidth={1.5} className="text-[var(--accent-highlight)]" />
        <div>
          <h1 className="text-2xl font-display font-bold text-[var(--text-primary)] m-0">
            Brain Analysis
          </h1>
          <p className="text-sm text-[var(--text-secondary)] m-0 mt-0.5">
            Holographic 3D neural visualization
          </p>
        </div>
      </div>

      {/* 3D Viewport */}
      <GlassCard className="h-[480px] p-0 overflow-hidden relative">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#8B5CF6]/10 via-transparent to-transparent pointer-events-none" />
        <Scene interactive={true} key={zoomKey} />
      </GlassCard>

      {/* Control Strip */}
      <div className="grid grid-cols-3 gap-3">
        <button 
          onClick={() => setAutoRotate(prev => !prev)}
          className="text-left w-full focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)] rounded-xl"
        >
          <GlassCard interactive className={`flex items-center gap-3 p-3 transition-colors ${autoRotate ? 'border-[var(--accent-primary)]/50 bg-[var(--accent-primary)]/10' : ''}`}>
            <RotateCw size={16} strokeWidth={1.5} className={autoRotate ? 'text-[var(--accent-primary)] animate-spin-slow' : 'text-[var(--text-secondary)]'} />
            <span className={`text-xs font-medium ${autoRotate ? 'text-[var(--accent-primary)]' : 'text-[var(--text-secondary)]'}`}>
              {autoRotate ? 'Rotate: On' : 'Rotate: Off'}
            </span>
          </GlassCard>
        </button>

        <button 
          onClick={() => setZoomKey(prev => prev + 1)}
          className="text-left w-full focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)] rounded-xl"
        >
          <GlassCard interactive className="flex items-center gap-3 p-3">
            <ZoomIn size={16} strokeWidth={1.5} className="text-[var(--text-secondary)]" />
            <span className="text-xs text-[var(--text-secondary)]">Reset View</span>
          </GlassCard>
        </button>

        <button 
          onClick={() => setHighlightRegion(prev => !prev)}
          className="text-left w-full focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)] rounded-xl"
        >
          <GlassCard interactive className={`flex items-center gap-3 p-3 transition-colors ${highlightRegion ? 'border-[var(--accent-highlight)]/50 bg-[var(--accent-highlight)]/10' : ''}`}>
            <Crosshair size={16} strokeWidth={1.5} className={highlightRegion ? 'text-[var(--accent-highlight)]' : 'text-[var(--text-secondary)]'} />
            <span className={`text-xs font-medium ${highlightRegion ? 'text-[var(--accent-highlight)]' : 'text-[var(--text-secondary)]'}`}>
              {highlightRegion ? 'Highlight: Active' : 'Highlight Region'}
            </span>
          </GlassCard>
        </button>
      </div>
    </motion.div>
  );
}
