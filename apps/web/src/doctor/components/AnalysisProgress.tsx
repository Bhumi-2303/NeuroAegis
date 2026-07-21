import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrainCircuit, Database, Activity, Network, ShieldCheck, PieChart } from 'lucide-react';

interface AnalysisProgressProps {
  jobId: string;
  onComplete: () => void;
}

interface JobStatus {
  status: string;
  progress: number;
}

export function AnalysisProgress({ jobId, onComplete }: AnalysisProgressProps) {
  const [status, setStatus] = useState<JobStatus>({ status: 'Connecting to AI Engine...', progress: 0 });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/v2/predict/status/${jobId}`);
        if (!res.ok) throw new Error("Failed to fetch status");
        const data = await res.json();
        
        setStatus({ status: data.status, progress: data.progress });
        
        if (data.status === 'Completed' || data.progress === 100) {
          clearInterval(pollInterval);
          setTimeout(onComplete, 1500); // Give time for the 100% animation to finish
        } else if (data.status.startsWith('Failed')) {
          clearInterval(pollInterval);
          setError(data.status);
        }
      } catch (e) {
        console.error(e);
      }
    }, 1000);

    return () => clearInterval(pollInterval);
  }, [jobId, onComplete]);

  const stages = [
    { name: 'Validating Patient Data', icon: Database, threshold: 10 },
    { name: 'Feature Extraction & Signal Processing', icon: Activity, threshold: 25 },
    { name: 'Brain Graph Construction', icon: Network, threshold: 45 },
    { name: 'Graph Neural Network Inference', icon: BrainCircuit, threshold: 65 },
    { name: 'Explainable AI (SHAP)', icon: PieChart, threshold: 85 },
    { name: 'Confidence Calculation', icon: ShieldCheck, threshold: 95 },
  ];

  const currentStageIndex = stages.findIndex(s => status.progress <= s.threshold);
  const activeStage = currentStageIndex === -1 ? stages.length - 1 : currentStageIndex;

  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center max-w-3xl mx-auto w-full">
      {error ? (
        <div className="bg-red-500/10 border border-red-500/50 rounded-2xl p-8 text-center text-red-400">
          <h3 className="text-xl font-bold mb-2">Analysis Failed</h3>
          <p>{error}</p>
        </div>
      ) : (
        <div className="w-full space-y-16">
          {/* Main Hero Animation */}
          <div className="relative flex justify-center items-center h-48">
            <motion.div
              animate={{ 
                scale: [1, 1.2, 1],
                rotate: [0, 180, 360]
              }}
              transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
              className="absolute w-48 h-48 border-[1px] border-dashed border-teal-500/30 rounded-full"
            />
            <motion.div
              animate={{ 
                scale: [1, 1.1, 1],
                rotate: [360, 180, 0]
              }}
              transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
              className="absolute w-64 h-64 border-[1px] border-dashed border-blue-500/20 rounded-full"
            />
            
            <AnimatePresence mode="wait">
              <motion.div
                key={activeStage}
                initial={{ opacity: 0, scale: 0.5, filter: "blur(10px)" }}
                animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                exit={{ opacity: 0, scale: 1.5, filter: "blur(10px)" }}
                transition={{ duration: 0.5 }}
                className="relative z-10 flex flex-col items-center"
              >
                {React.createElement(stages[Math.min(activeStage, stages.length - 1)].icon, { 
                  className: "w-20 h-20 text-teal-400 mb-6 drop-shadow-[0_0_15px_rgba(45,212,191,0.5)]" 
                })}
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Progress Status */}
          <div className="text-center space-y-6">
            <h2 className="text-3xl font-light tracking-wide bg-gradient-to-r from-teal-400 to-blue-400 bg-clip-text text-transparent">
              {status.status}
            </h2>
            
            <div className="w-full bg-black/40 rounded-full h-3 border border-white/10 overflow-hidden relative">
              <motion.div
                className="absolute left-0 top-0 h-full bg-gradient-to-r from-teal-500 to-blue-500 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${status.progress}%` }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              />
              <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9InAiIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTTAgNDBsNDAtNDBINjBMMjAgNDBIMHoiIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iLjA1Ii8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI3ApIi8+PC9zdmc+')] opacity-20 animate-[slide_2s_linear_infinite]" />
            </div>
          </div>

          {/* Cinematic Pipeline Steps */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {stages.map((stage, index) => {
              const isPast = activeStage > index || status.progress === 100;
              const isActive = activeStage === index && status.progress < 100;
              
              return (
                <div 
                  key={index}
                  className={`relative p-4 rounded-xl border backdrop-blur-sm transition-all duration-500 ${
                    isActive ? 'bg-teal-500/10 border-teal-500/50 shadow-[0_0_20px_rgba(45,212,191,0.15)] scale-105' :
                    isPast ? 'bg-white/5 border-white/10 opacity-70' :
                    'bg-black/20 border-white/5 opacity-30 grayscale'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${isActive ? 'bg-teal-500/20 text-teal-400' : isPast ? 'bg-white/10 text-white' : 'bg-black/40 text-neutral-500'}`}>
                      {React.createElement(stage.icon, { className: "w-4 h-4" })}
                    </div>
                    <p className={`text-sm font-medium ${isActive ? 'text-teal-400' : 'text-neutral-300'}`}>
                      {stage.name}
                    </p>
                  </div>
                  {isActive && (
                    <motion.div 
                      layoutId="active-indicator"
                      className="absolute -bottom-px left-4 right-4 h-[2px] bg-gradient-to-r from-transparent via-teal-400 to-transparent" 
                    />
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
      <style>{`
        @keyframes slide {
          from { background-position: 0 0; }
          to { background-position: 40px 0; }
        }
      `}</style>
    </div>
  );
}
