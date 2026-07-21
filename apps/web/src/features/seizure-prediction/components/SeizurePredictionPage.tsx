import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Upload, FileText, Activity, CheckCircle, Clock } from 'lucide-react';
import { 
  GlassCard, 
  EmptyState, 
  ErrorState, 
  ConfidenceGauge, 
  StatusBadge 
} from '../../../shared/components';
import { pageTransition, slideUp, staggerChildren, fadeIn } from '../../../shared/lib/motion-presets';
import type { ModelOutput } from '@neuroaegis/model-contracts';

const MAX_FILE_SIZE = 52 * 1024 * 1024; // 52MB

type JobStep = 'Validating' | 'Preprocessing' | 'Inference' | 'SHAP';
const STEPS: JobStep[] = ['Validating', 'Preprocessing', 'Inference', 'SHAP'];

export const SeizurePredictionPage: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<'random_forest' | 'xgboost' | 'lightgbm'>('random_forest');
  const [file, setFile] = useState<File | null>(null);
  const [samplingRate, setSamplingRate] = useState<number>(256);
  const [channels, setChannels] = useState<string>('EEG-Fpz-Cz, EEG-Pz-Oz');
  const [validationError, setValidationError] = useState<string | null>(null);
  
  const [isUploading, setIsUploading] = useState(false);
  const [currentStep, setCurrentStep] = useState<JobStep | null>(null);
  const [data, setData] = useState<ModelOutput | null>(null);
  const [isError, setIsError] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.size > MAX_FILE_SIZE) {
        setValidationError(`File exceeds 52MB limit (${(selectedFile.size / 1024 / 1024).toFixed(1)}MB)`);
        setFile(null);
      } else {
        setValidationError(null);
        setFile(selectedFile);
      }
    }
  };

  const handlePredict = async () => {
    if (!file) {
      setValidationError("Please select an EEG file to analyze.");
      return;
    }

    setIsUploading(true);
    setCurrentStep('Validating');
    setIsError(false);
    setData(null);

    try {
      // Mock job polling sequence to mimic /api/predict/status/{job_id}
      await new Promise(r => setTimeout(r, 800));
      setCurrentStep('Preprocessing');
      
      await new Promise(r => setTimeout(r, 1200));
      setCurrentStep('Inference');
      
      await new Promise(r => setTimeout(r, 1500));
      setCurrentStep('SHAP');
      
      await new Promise(r => setTimeout(r, 1000));
      
      // Return final output
      const responseData: ModelOutput = {
        modelName: selectedModel,
        prediction: {
          label: Math.random() > 0.5 ? 'seizure' : 'non_seizure',
          probabilities: { seizure: 0.85, non_seizure: 0.15 }
        },
        confidence: { value: 0.88, band: 'high' },
        explanation: {
          baseValue: 0.35,
          features: [
            { featureName: 'Alpha Power', value: 0.12 },
            { featureName: 'Beta Power', value: -0.05 },
            { featureName: 'Theta Peak', value: 0.08 }
          ]
        },
        generatedAt: new Date().toISOString()
      };

      setData(responseData);
    } catch (err) {
      console.error(err);
      setIsError(true);
    } finally {
      setIsUploading(false);
      setCurrentStep(null);
    }
  };

  return (
    <motion.div {...pageTransition} className="p-6 space-y-6">
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-8 h-8 text-[var(--accent-primary)]" strokeWidth={1.5} />
        <h1 className="text-2xl font-[var(--font-display)] text-[var(--text-primary)]">Seizure Prediction</h1>
      </div>

      <motion.div {...staggerChildren} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Form Section */}
        <motion.div {...slideUp} className="lg:col-span-1">
          <GlassCard title="EEG Upload" className="p-6">
            <div className="space-y-5">
              
              {/* File Upload Area */}
              <div>
                <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-2">EEG File (.csv)</label>
                <div 
                  className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${file ? 'border-[var(--accent-primary)] bg-[var(--accent-primary)]/10' : 'border-[var(--bg-3)] hover:border-[var(--accent-secondary)] hover:bg-[var(--bg-2)]'}`}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input 
                    type="file" 
                    accept=".csv,.txt"
                    ref={fileInputRef} 
                    className="hidden" 
                    onChange={handleFileChange}
                  />
                  {file ? (
                    <div className="flex flex-col items-center gap-2">
                      <FileText className="w-8 h-8 text-[var(--accent-primary)]" />
                      <span className="text-sm font-[var(--font-body)] text-[var(--text-primary)]">{file.name}</span>
                      <span className="text-xs text-[var(--text-secondary)]">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-2">
                      <Upload className="w-8 h-8 text-[var(--text-secondary)]" />
                      <span className="text-sm font-[var(--font-body)] text-[var(--text-secondary)]">Click to upload or drag and drop</span>
                      <span className="text-xs text-[var(--text-secondary)] opacity-70">Max size 52MB</span>
                    </div>
                  )}
                </div>
                {validationError && (
                  <p className="mt-2 text-xs text-[var(--state-danger)] font-[var(--font-body)]">{validationError}</p>
                )}
              </div>

              {/* Form Controls */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-1">Sampling Rate (Hz)</label>
                  <input 
                    type="number" 
                    value={samplingRate}
                    onChange={e => setSamplingRate(Number(e.target.value))}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-secondary)]"
                  />
                </div>
                <div>
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-1">Model</label>
                  <select 
                    value={selectedModel}
                    onChange={e => setSelectedModel(e.target.value as any)}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-secondary)]"
                  >
                    <option value="random_forest">Random Forest</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="lightgbm">LightGBM</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-1">Channels (comma separated)</label>
                <input 
                  type="text" 
                  value={channels}
                  onChange={e => setChannels(e.target.value)}
                  className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-secondary)]"
                  placeholder="e.g. EEG-Fpz-Cz"
                />
              </div>

              <button 
                onClick={handlePredict}
                disabled={isUploading || !file}
                className="w-full py-3 rounded-lg bg-[var(--accent-primary)] text-[var(--bg-1)] font-[var(--font-body)] font-medium text-sm flex items-center justify-center gap-2 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {isUploading ? <Activity className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
                {isUploading ? 'Analyzing...' : 'Run Prediction'}
              </button>
            </div>
          </GlassCard>
        </motion.div>

        {/* Results Section */}
        <motion.div {...slideUp} className="lg:col-span-2 flex flex-col gap-6">
          {isError && !isUploading && (
            <ErrorState 
              title="Prediction Failed" 
              message="Unable to generate prediction. Please check your file and try again." 
              onRetry={handlePredict} 
              retryLabel="Retry" 
            />
          )}

          {!isUploading && !isError && !data && (
            <EmptyState 
              icon={Activity} 
              title="Ready for Analysis" 
              description="Upload an EEG file and configure parameters to run prediction." 
            />
          )}

          {isUploading && (
            <GlassCard title="Analysis in Progress" className="p-8 h-full flex flex-col justify-center">
              <div className="max-w-md mx-auto w-full space-y-6">
                {STEPS.map((step, idx) => {
                  const isActive = step === currentStep;
                  const isPast = STEPS.indexOf(currentStep!) > idx;
                  
                  return (
                    <div key={step} className={`flex items-center gap-4 transition-all duration-300 ${isActive ? 'opacity-100 scale-105' : 'opacity-50'}`}>
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 border ${
                        isPast ? 'bg-[var(--state-success)]/20 border-[var(--state-success)] text-[var(--state-success)]' :
                        isActive ? 'bg-[var(--accent-primary)]/20 border-[var(--accent-primary)] text-[var(--accent-primary)]' :
                        'bg-[var(--bg-3)] border-[var(--bg-3)] text-[var(--text-secondary)]'
                      }`}>
                        {isPast ? <CheckCircle className="w-4 h-4" /> : isActive ? <Activity className="w-4 h-4 animate-spin" /> : <Clock className="w-4 h-4" />}
                      </div>
                      <div className="flex-1">
                        <h4 className={`text-sm font-[var(--font-body)] font-medium ${isActive || isPast ? 'text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}`}>
                          {step}
                        </h4>
                        {isActive && (
                          <div className="w-full bg-[var(--bg-3)] rounded-full h-1 mt-2 overflow-hidden">
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
          )}

          {!isUploading && !isError && data && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-full">
              <GlassCard title="Model Confidence" className="flex flex-col items-center justify-center p-8">
                <ConfidenceGauge 
                  value={data.confidence.value * 100} 
                  size={180} 
                  label={`Band: ${data.confidence.band.toUpperCase()}`} 
                />
                <div className="mt-8">
                  <StatusBadge 
                    status={data.prediction.label === 'seizure' ? 'critical' : 'normal'} 
                    label={data.prediction.label === 'seizure' ? 'Seizure Detected' : 'Normal Activity'} 
                  />
                </div>
              </GlassCard>

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
                  <span>Model: {data.modelName}</span>
                  <span>Generated: {new Date(data.generatedAt).toLocaleTimeString()}</span>
                </div>
              </GlassCard>
            </div>
          )}
        </motion.div>
      </motion.div>
    </motion.div>
  );
};


