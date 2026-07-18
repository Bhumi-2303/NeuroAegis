
import { motion } from 'framer-motion';
import { Settings, SlidersHorizontal } from 'lucide-react';
import { GlassCard, EmptyState } from '../../../shared/components';
import { pageTransition } from '../../../shared/lib/motion-presets';

export const SettingsPage = () => {
  return (
    <motion.div variants={pageTransition} initial="initial" animate="animate" exit="exit" className="p-6 max-w-6xl mx-auto space-y-6">
      <header className="flex items-center gap-3 mb-8">
        <Settings className="w-8 h-8 text-[var(--accent-primary)]" />
        <h1 className="text-3xl font-display font-bold text-[var(--text-primary)]">Settings</h1>
      </header>

      <GlassCard title="Application Configuration">
        <div className="p-8 border-t border-[var(--bg-3)]">
          <EmptyState
            icon={SlidersHorizontal}
            title="Configuration Not Available"
            description="Configuration options will be available in a future update."
          />
        </div>
      </GlassCard>
    </motion.div>
  );
};
