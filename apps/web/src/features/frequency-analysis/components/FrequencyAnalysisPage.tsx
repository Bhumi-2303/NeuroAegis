import React from 'react';
import { motion } from 'framer-motion';
import { Radio, Lock } from 'lucide-react';

import { pageTransition } from '../../../shared/lib/motion-presets';

export const FrequencyAnalysisPage: React.FC = () => {
  const renderContent = () => {
    return (
      <div className="flex flex-col items-center justify-center h-[500px] text-center border-2 border-dashed border-[var(--bg-4)] rounded-xl bg-[var(--bg-2)]/50">
        <Lock className="text-gray-500 mb-3" size={32} />
        <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2">Awaiting Backend Integration</h3>
        <p className="text-sm text-[var(--text-secondary)] max-w-[400px]">
          Frequency band analysis and spectral power streams will be available when live EEG processing is implemented on the backend.
        </p>
      </div>
    );
  };

  return (
    <motion.div 
      className="p-6 h-full flex flex-col"
      variants={pageTransition}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      <header className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-[var(--bg-2)] border border-[rgba(255,255,255,0.1)] rounded-lg">
          <Radio className="w-6 h-6 text-[var(--accent-primary)]" />
        </div>
        <h1 className="text-2xl font-display text-[var(--text-primary)]">Frequency Band Analysis</h1>
      </header>

      <main className="flex-1">
        {renderContent()}
      </main>
    </motion.div>
  );
};

export default FrequencyAnalysisPage;
