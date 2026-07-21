import { useState } from 'react';
import { PatientForm } from '../components/PatientForm';
import { AnalysisProgress } from '../components/AnalysisProgress';
import { ResultDashboard } from '../components/ResultDashboard';
import { motion, AnimatePresence } from 'framer-motion';

type DashboardStage = 'input' | 'analyzing' | 'result';

export function DoctorDashboard() {
  const [stage, setStage] = useState<DashboardStage>('input');
  const [jobId, setJobId] = useState<string | null>(null);

  const handleAnalysisStart = (newJobId: string) => {
    setJobId(newJobId);
    setStage('analyzing');
  };

  const handleAnalysisComplete = () => {
    setStage('result');
  };

  const handleReset = () => {
    setJobId(null);
    setStage('input');
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-white font-sans overflow-x-hidden selection:bg-teal-500/30">
      {/* Animated background elements */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-teal-900/20 blur-[120px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-blue-900/20 blur-[120px]" />
      </div>

      <header className="relative z-10 p-6 border-b border-white/10 backdrop-blur-md bg-neutral-950/50 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-teal-500 to-blue-500 flex items-center justify-center font-bold">N</div>
          <h1 className="text-xl font-medium tracking-wide">NeuroAegis <span className="text-neutral-400">Doctor AI</span></h1>
        </div>
        {stage !== 'input' && (
          <button 
            onClick={handleReset}
            className="px-4 py-2 text-sm text-neutral-400 hover:text-white transition-colors"
          >
            New Patient
          </button>
        )}
      </header>

      <main className="relative z-10 container mx-auto px-4 py-8 max-w-7xl">
        <AnimatePresence mode="wait">
          {stage === 'input' && (
            <motion.div
              key="input"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.4 }}
            >
              <PatientForm onStartAnalysis={handleAnalysisStart} />
            </motion.div>
          )}

          {stage === 'analyzing' && jobId && (
            <motion.div
              key="analyzing"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              transition={{ duration: 0.4 }}
            >
              <AnalysisProgress jobId={jobId} onComplete={handleAnalysisComplete} />
            </motion.div>
          )}

          {stage === 'result' && jobId && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <ResultDashboard jobId={jobId} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
