import { motion } from 'framer-motion';
import { Brain, RotateCw, ZoomIn, Crosshair } from 'lucide-react';
import { GlassCard, Scene } from '../../../shared/components';
import { slideUp } from '../../../shared/lib/motion-presets';

export function BrainAnalysisPage(): React.JSX.Element {
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
        <Scene interactive={true} />
      </GlassCard>

      {/* Control Strip */}
      <div className="grid grid-cols-3 gap-3">
        <GlassCard className="flex items-center gap-3 p-3 opacity-50 cursor-not-allowed">
          <RotateCw size={16} strokeWidth={1.5} className="text-[var(--text-secondary)]" />
          <span className="text-xs text-[var(--text-secondary)]">Rotate</span>
        </GlassCard>
        <GlassCard className="flex items-center gap-3 p-3 opacity-50 cursor-not-allowed">
          <ZoomIn size={16} strokeWidth={1.5} className="text-[var(--text-secondary)]" />
          <span className="text-xs text-[var(--text-secondary)]">Zoom</span>
        </GlassCard>
        <GlassCard className="flex items-center gap-3 p-3 opacity-50 cursor-not-allowed">
          <Crosshair size={16} strokeWidth={1.5} className="text-[var(--text-secondary)]" />
          <span className="text-xs text-[var(--text-secondary)]">Highlight Region</span>
        </GlassCard>
      </div>
    </motion.div>
  );
}
