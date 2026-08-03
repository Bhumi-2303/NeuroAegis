import React, { useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Upload, FileText, Activity, CheckCircle, Clock } from 'lucide-react';
import { 
  GlassCard, 

  ErrorState, 
  ConfidenceGauge, 
  StatusBadge,
  WaveformSparkline
} from '../../../shared/components';
import { pageTransition, staggerChildren, fadeIn } from '../../../shared/lib/motion-presets';
import { usePredictionFlow, STEPS } from '../../../shared/hooks';
import { ShapWaterfall } from '../../explainability/components/ShapWaterfall';

export function DashboardPage(): React.JSX.Element {
  const {
    file,

    samplingRate,
    setSamplingRate,
    channels,
    setChannels,
    validationError,

    isUploading,
    currentStep,
    data,
    isError,
    errorMessage,
    datasetName,
    detectionConfidence,
    modelName,
    handleFileChange,
    resetAnalysis,
    handlePredict
  } = usePredictionFlow();

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Since we don't have the real uploaded data available immediately for the waveform,
  // we'll simulate a static segment representing the analyzed signal.
  const mockStaticWaveform = Array.from({ length: 100 }, (_, i) => Math.sin(i * 0.5) * 50 + (Math.random() * 20 - 10));

  const [rawSignal, setRawSignal] = React.useState<number[]>([]);

  React.useEffect(() => {
    if (file && data) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        if (content) {
          const lines = content.split('\n');
          const vals: number[] = [];
          for (const line of lines) {
             const val = parseFloat(line.trim());
             if (!isNaN(val)) vals.push(val);
             if (vals.length >= 1000) break; // Limit to 1000 samples for the waveform
          }
          setRawSignal(vals.length > 0 ? vals : mockStaticWaveform);
        }
      };
      reader.readAsText(file);
    } else {
      setRawSignal([]);
    }
  }, [file, data]);

  return (
    <motion.div {...pageTransition} className="flex flex-col gap-6">
      {/* Page Title */}
      <div className="flex items-center gap-3">
        <Brain size={28} strokeWidth={1.5} className="text-[var(--accent-primary)]" />
        <div>
          <h1 id="dashboard-title" className="text-2xl font-display font-bold text-[var(--text-primary)] m-0">
            NeuroAegis
          </h1>
          <p className="text-sm text-[var(--text-secondary)] m-0 mt-0.5">
            Seizure detection dashboard.
          </p>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {/* Idle State: File Upload Form + Scene Hero */}
        {!isUploading && !isError && !data && (
          <motion.div key="idle" {...fadeIn} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Upload Form */}
            <GlassCard title="EEG Upload" className="p-8 lg:col-span-1 h-fit">
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-2">EEG File (.csv, .txt, .edf)</label>
                  <div 
                    className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${file ? 'border-[var(--accent-primary)] bg-[var(--accent-primary)]/10' : 'border-[var(--bg-3)] hover:border-[var(--accent-secondary)] hover:bg-[var(--bg-2)]'}`}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input 
                      type="file" 
                      accept=".csv,.txt,.edf"
                      ref={fileInputRef} 
                      className="hidden" 
                      onChange={handleFileChange}
                    />
                    {file ? (
                      <div className="flex flex-col items-center gap-2">
                        <FileText className="w-10 h-10 text-[var(--accent-primary)]" />
                        <span className="text-sm font-[var(--font-body)] text-[var(--text-primary)]">{file.name}</span>
                        <span className="text-xs text-[var(--text-secondary)]">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-3">
                        <Upload className="w-10 h-10 text-[var(--text-secondary)]" />
                        <span className="text-sm font-[var(--font-body)] text-[var(--text-secondary)]">Click to upload or drag and drop</span>
                        <span className="text-xs text-[var(--text-secondary)] opacity-70">Max size 52MB</span>
                      </div>
                    )}
                  </div>
                  {validationError && (
                    <p className="mt-2 text-xs text-[var(--state-danger)] font-[var(--font-body)]">{validationError}</p>
                  )}
                </div>

                <div className="grid grid-cols-1 gap-4">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="block text-sm font-medium text-[var(--text-secondary)]">Sampling Rate (Hz)</label>
                      <span className="text-xs text-[var(--text-secondary)] opacity-70">Optional</span>
                    </div>
                    <input 
                      type="number" 
                      placeholder="e.g. 256"
                      value={samplingRate}
                      onChange={e => setSamplingRate(e.target.value)}
                      className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-3 py-3 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-secondary)]"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-1">Channels (comma separated)</label>
                  <input 
                    type="text" 
                    value={channels}
                    onChange={e => setChannels(e.target.value)}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-3 py-3 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-secondary)]"
                    placeholder="e.g. EEG-Fpz-Cz"
                  />
                </div>

                <button 
                  onClick={handlePredict}
                  disabled={isUploading || !file}
                  className="w-full py-4 rounded-lg bg-[var(--accent-primary)] text-[var(--bg-1)] font-[var(--font-body)] font-medium text-sm flex items-center justify-center gap-2 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all mt-4"
                >
                  <Brain className="w-5 h-5" />
                  Run Prediction
                </button>
              </div>
            </GlassCard>

            {/* Instructions Area */}
            <GlassCard
              className="lg:col-span-2 min-h-[400px] h-full p-8 flex flex-col justify-center items-center text-center bg-[var(--bg-2)] border-[var(--bg-3)]"
              role="region"
              aria-label="Instructions"
            >
              <div className="flex flex-col items-center gap-4 max-w-md">
                 <div className="w-16 h-16 rounded-2xl bg-[var(--bg-3)] flex items-center justify-center text-[var(--accent-primary)]">
                   <Brain size={32} strokeWidth={1.5} />
                 </div>
                 <h2 className="text-xl font-semibold text-[var(--text-primary)]">Ready for Analysis</h2>
                 <p className="text-sm text-[var(--text-secondary)]">Upload an EEG file containing neural data to begin the prediction pipeline. The system will process the signals and evaluate seizure likelihood.</p>
              </div>
            </GlassCard>
            
          </motion.div>
        )}

        {/* Loading State */}
        {isUploading && (
          <motion.div key="loading" {...fadeIn} className="max-w-xl mx-auto mt-20">
            <GlassCard title="Analysis in Progress" className="p-10 flex flex-col justify-center">
              <div className="w-full space-y-8">
                {STEPS.map((step, idx) => {
                  const isActive = step === currentStep;
                  const isPast = STEPS.indexOf(currentStep!) > idx;
                  
                  return (
                    <div key={step} className={`flex items-center gap-5 transition-all duration-300 ${isActive ? 'opacity-100 scale-105' : 'opacity-50'}`}>
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 border-2 ${
                        isPast ? 'bg-[var(--state-success)]/20 border-[var(--state-success)] text-[var(--state-success)]' :
                        isActive ? 'bg-[var(--accent-primary)]/20 border-[var(--accent-primary)] text-[var(--accent-primary)]' :
                        'bg-[var(--bg-3)] border-[var(--bg-3)] text-[var(--text-secondary)]'
                      }`}>
                        {isPast ? <CheckCircle className="w-5 h-5" /> : isActive ? <Activity className="w-5 h-5 animate-spin" /> : <Clock className="w-5 h-5" />}
                      </div>
                      <div className="flex-1">
                        <h4 className={`text-base font-[var(--font-body)] font-medium ${isActive || isPast ? 'text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}`}>
                          {step}
                        </h4>
                        {isActive && (
                          <div className="w-full bg-[var(--bg-3)] rounded-full h-1 mt-3 overflow-hidden">
                            <motion.div 
                              className="bg-[var(--accent-primary)] h-1 rounded-full" 
                              initial={{ width: 0 }}
                              animate={{ width: "100%" }}
                              transition={{ duration: 10, ease: "linear" }}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </GlassCard>
          </motion.div>
        )}

        {/* Error State */}
        {isError && !isUploading && (
          <motion.div key="error" {...fadeIn} className="max-w-2xl mx-auto mt-20">
            <ErrorState 
              title="Prediction Failed" 
              message={errorMessage || "Unable to generate prediction. Please check your file and try again."} 
              onRetry={resetAnalysis} 
              retryLabel="Start Over" 
            />
          </motion.div>
        )}

        {/* Results State */}
        {!isUploading && !isError && data && (
          <motion.div key="results" {...staggerChildren} className="flex flex-col gap-6">
            <div className="flex justify-between items-center bg-[var(--bg-2)] p-4 rounded-xl border border-[var(--bg-3)]">
              <div className="flex flex-col">
                <span className="text-xs text-[var(--text-secondary)] font-[var(--font-mono)] uppercase tracking-wider">Analysis Complete</span>
                <span className="text-sm font-[var(--font-body)] text-[var(--text-primary)]">{file?.name}</span>
              </div>
              <button 
                onClick={resetAnalysis}
                className="px-5 py-2.5 rounded-lg bg-[var(--bg-3)] text-[var(--text-primary)] font-[var(--font-body)] text-sm flex items-center gap-2 hover:bg-[var(--bg-3)]/80 hover:text-[var(--accent-primary)] transition-all border border-[var(--bg-3)] shadow-sm"
              >
                <Upload className="w-4 h-4" />
                Upload Another File
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
              {/* Confidence Gauge */}
              <GlassCard title="Model Confidence" className="flex flex-col items-center justify-center p-8">
                <ConfidenceGauge 
                  value={Math.max(data.prediction.probabilities.seizure, data.prediction.probabilities.non_seizure) * 100} 
                  size={180} 
                />
                <div className="mt-8">
                  <StatusBadge 
                    status={data.prediction.label === 'seizure' ? 'critical' : 'normal'} 
                    label={data.prediction.label === 'seizure' ? 'Seizure Detected' : 'Normal Activity'} 
                  />
                </div>
              </GlassCard>



              {/* Prediction Probability */}
              <GlassCard title="Probability Breakdown" className="p-6 flex flex-col justify-center gap-6">
                <motion.div {...fadeIn}>
                  <div className="flex justify-between mb-2">
                    <span className="font-[var(--font-body)] text-[var(--text-secondary)]">Seizure</span>
                    <span className="font-[var(--font-mono)] text-[var(--text-primary)]">
                      {(data.prediction.probabilities.seizure * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-[var(--bg-3)] rounded-full h-2">
                    <div 
                      className="bg-[var(--state-danger)] h-2 rounded-full transition-all duration-1000" 
                      style={{ width: `${data.prediction.probabilities.seizure * 100}%` }}
                    />
                  </div>
                </motion.div>

                <motion.div {...fadeIn}>
                  <div className="flex justify-between mb-2">
                    <span className="font-[var(--font-body)] text-[var(--text-secondary)]">Non-Seizure</span>
                    <span className="font-[var(--font-mono)] text-[var(--text-primary)]">
                      {(data.prediction.probabilities.non_seizure * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-[var(--bg-3)] rounded-full h-2">
                    <div 
                      className="bg-[var(--state-success)] h-2 rounded-full transition-all duration-1000" 
                      style={{ width: `${data.prediction.probabilities.non_seizure * 100}%` }}
                    />
                  </div>
                </motion.div>
                
                <div className="mt-4 p-4 bg-[var(--bg-2)] rounded-xl border border-[var(--bg-3)] text-xs text-[var(--text-secondary)] font-[var(--font-mono)] flex flex-col gap-1">
                  <div className="grid grid-cols-2 gap-2 mb-2">
                    <div>
                      <span className="block text-[var(--text-secondary)]">Sampling Rate</span>
                      <span className="block text-[var(--text-primary)]">{samplingRate || 'Auto-detected'}</span>
                    </div>
                    <div>
                      <span className="block text-[var(--text-secondary)]">Channels</span>
                      <span className="block text-[var(--text-primary)]">{channels ? channels.split(',').length : 'Auto-detected'}</span>
                    </div>
                  </div>
                  <span>Dataset: {datasetName?.toUpperCase()} (Conf: {(detectionConfidence! * 100).toFixed(1)}%)</span>
                  <span>Model: {modelName?.replace('_', ' ').toUpperCase()}</span>
                  <span>Generated: {new Date(data.generatedAt).toLocaleTimeString()}</span>
                </div>
              </GlassCard>
            </div>

            {/* Explainability and Waveform Section */}
            {data.explanation && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full mt-6">
                <ShapWaterfall 
                  baseValue={data.explanation.baseValue}
                  features={data.explanation.features || []}
                  finalProbability={data.prediction.probabilities.seizure}
                />
                
                <GlassCard title="Analyzed signal segment" className="p-6 h-full flex flex-col justify-center">
                  <div className="h-64 w-full">
                     <WaveformSparkline data={rawSignal.length > 0 ? rawSignal : mockStaticWaveform} color="var(--accent-primary)" />
                  </div>
                </GlassCard>
              </div>
            )}

          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
