import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, CircleDashed } from 'lucide-react';
import { WidgetCard } from './WidgetCard';

const PIPELINE_STAGES = [
  'Loading EEG Signal Data',
  'Filtering & Artifact Removal',
  'Extracting Biomarkers & Features',
  'Running Model Inference',
  'Generating SHAP Explanations',
  'Compiling Clinical Report'
];

export function LoadingPipeline() {
  const [currentStage, setCurrentStage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentStage(prev => {
        if (prev < PIPELINE_STAGES.length - 1) return prev + 1;
        clearInterval(timer);
        return prev;
      });
    }, 1200); // Move stage every 1.2s for demo purposes

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="max-w-2xl mx-auto mt-16 w-full px-4">
      <WidgetCard>
        <div className="p-8">
          <div className="flex flex-col items-center mb-10">
            <div className="text-xl font-bold text-[var(--text-primary)] mb-2">Analysis in Progress</div>
            <div className="text-sm text-[var(--text-secondary)]">Processing biomedical data using remote cluster...</div>
          </div>

          <div className="space-y-6">
            {PIPELINE_STAGES.map((stage, idx) => {
              const isPast = currentStage > idx;
              const isActive = currentStage === idx;
              const isFuture = currentStage < idx;

              return (
                <div key={stage} className={`flex items-center gap-4 transition-all duration-500 ${isActive ? 'scale-[1.02] opacity-100' : isPast ? 'opacity-70' : 'opacity-40'}`}>
                  
                  <div className="shrink-0 relative flex items-center justify-center w-8 h-8">
                    {isPast && <CheckCircle2 size={24} className="text-emerald-500" />}
                    {isActive && (
                      <motion.div 
                        animate={{ rotate: 360 }}
                        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                      >
                        <CircleDashed size={24} className="text-blue-500" />
                      </motion.div>
                    )}
                    {isFuture && <div className="w-4 h-4 rounded-full border-2 border-[var(--bg-4)]" />}
                  </div>

                  <div className="flex-1">
                    <div className={`text-sm font-semibold ${isActive ? 'text-blue-600' : isPast ? 'text-[var(--text-primary)]' : 'text-[var(--text-muted)]'}`}>
                      {stage}
                    </div>
                    {isActive && (
                      <div className="w-full h-1 bg-[var(--bg-3)] rounded-full mt-2 overflow-hidden">
                        <motion.div 
                          className="h-full bg-blue-500 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: '100%' }}
                          transition={{ duration: 1.2, ease: 'linear' }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </WidgetCard>
    </div>
  );
}
