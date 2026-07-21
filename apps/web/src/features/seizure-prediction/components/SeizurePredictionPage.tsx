import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Upload, FileText, Activity, CheckCircle, Clock } from 'lucide-react';
import { 
  GlassCard, 

  ErrorState, 
  ConfidenceGauge, 
  StatusBadge 
} from '../../../shared/components';
import { pageTransition, staggerChildren, fadeIn } from '../../../shared/lib/motion-presets';
import type { ModelOutput } from '@neuroaegis/model-contracts';

const MAX_FILE_SIZE = 52 * 1024 * 1024; // 52MB

type JobStep = 'Validating' | 'Preprocessing' | 'Inference' | 'SHAP';
const STEPS: JobStep[] = ['Validating', 'Preprocessing', 'Inference', 'SHAP'];

interface AvailableModels {
  [dataset: string]: {
    loaded: boolean;
    models: string[];
    default_model: string;
    metadata: any;
  }
}

export const SeizurePredictionPage: React.FC = () => {
  const [availableModels, setAvailableModels] = useState<AvailableModels>({});
  const [datasetName, setDatasetName] = useState<string | null>(null);
  const [detectionConfidence, setDetectionConfidence] = useState<number | null>(null);
  const [modelName, setModelName] = useState<string | null>(null);
  
  const [file, setFile] = useState<File | null>(null);
  const [samplingRate, setSamplingRate] = useState<number>(256);
  const [channels, setChannels] = useState<string>('EEG-Fpz-Cz, EEG-Pz-Oz');
  const [validationError, setValidationError] = useState<string | null>(null);
  
  const [isUploading, setIsUploading] = useState(false);
  const [currentStep, setCurrentStep] = useState<JobStep | null>(null);
  const [data, setData] = useState<ModelOutput | null>(null);
  const [isError, setIsError] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch('/api/v1/models/');
        if (res.ok) {
          const data = await res.json();
          setAvailableModels(data);

        }
      } catch (err) {
        console.error("Failed to fetch available models", err);
      }
    };
    fetchModels();
  }, []);
  // Datasets are fetched, but we no longer pre-select or require them in standard workflow.
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
      const formData = new FormData();
      formData.append('file', file);
      formData.append('sampling_rate', samplingRate.toString());
      formData.append('channels', channels);
      // Not explicitly appending dataset and model so backend auto-detects
      // formData.append('dataset', selectedDataset);
      // formData.append('model', selectedModel);

      const res = await fetch('/api/v1/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) throw new Error('Prediction request failed');
      const { job_id } = await res.json();
      
      while (true) {
        const statusRes = await fetch(`/api/v1/jobs/${job_id}`);
        if (!statusRes.ok) throw new Error('Failed to fetch job status');
        const statusData = await statusRes.json();
        
        // Map backend stages to UI steps
        const s = statusData.status;
        if (s.includes('Validating')) setCurrentStep('Validating');
        else if (s.includes('Feature') || s.includes('Processing')) setCurrentStep('Preprocessing');
        else if (s.includes('Inference') || s.includes('Graph')) setCurrentStep('Inference');
        else if (s.includes('SHAP') || s.includes('Explainable') || s.includes('Confidence')) setCurrentStep('SHAP');
        
        if (statusData.status === 'Completed') {
          const responseData: ModelOutput = {
            modelName: statusData.modelName || 'Unknown',
            prediction: {
               label: statusData.result.prediction_label,
               probabilities: { 
                 seizure: statusData.result.probability_seizure,
                 non_seizure: 1.0 - statusData.result.probability_seizure
               }
            },
            confidence: {
               value: statusData.result.probability_seizure,
               band: statusData.result.confidence_band
            },
            explanation: statusData.result.shap_explanation,
            generatedAt: new Date().toISOString()
          };
          setDatasetName(statusData.datasetName || 'Unknown');
          setDetectionConfidence(statusData.detectionConfidence || 0);
          setModelName(statusData.modelName || 'Unknown');
          setData(responseData);
          break;
        } else if (statusData.status === 'Failed' || statusData.error) {
           throw new Error(statusData.error || 'Job failed');
        }
        
        await new Promise(r => setTimeout(r, 1000));
      }
    } catch (err) {
      console.error(err);
      setIsError(true);
    } finally {
      setIsUploading(false);
      setCurrentStep(null);
    }
  };

  const loadedDatasets = Object.keys(availableModels).filter(k => availableModels[k].loaded);

  const resetAnalysis = () => {
    setFile(null);
    setData(null);
    setIsError(false);
    setCurrentStep(null);
    setDatasetName(null);
    setDetectionConfidence(null);
    setModelName(null);
    setValidationError(null);
  };

  return (
    <motion.div {...pageTransition} className="p-6 space-y-6">
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-8 h-8 text-[var(--accent-primary)]" strokeWidth={1.5} />
        <h1 className="text-2xl font-[var(--font-display)] text-[var(--text-primary)]">Seizure Prediction</h1>
      </div>

      <AnimatePresence mode="wait">
        {/* Idle State: File Upload Form */}
        {!isUploading && !isError && !data && (
          <motion.div key="idle" {...fadeIn} className="max-w-2xl mx-auto mt-12">
            <GlassCard title="EEG Upload" className="p-8">
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-2">EEG File (.csv)</label>
                  <div 
                    className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${file ? 'border-[var(--accent-primary)] bg-[var(--accent-primary)]/10' : 'border-[var(--bg-3)] hover:border-[var(--accent-secondary)] hover:bg-[var(--bg-2)]'}`}
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
                    <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-1">Sampling Rate (Hz)</label>
                    <input 
                      type="number" 
                      value={samplingRate}
                      onChange={e => setSamplingRate(Number(e.target.value))}
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
                  disabled={isUploading || !file || loadedDatasets.length === 0}
                  className="w-full py-4 rounded-lg bg-[var(--accent-primary)] text-[var(--bg-1)] font-[var(--font-body)] font-medium text-sm flex items-center justify-center gap-2 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all mt-4"
                >
                  <Brain className="w-5 h-5" />
                  Run Prediction
                </button>
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
              message="Unable to generate prediction. Please check your file and try again." 
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
                  <span>Dataset: {datasetName?.toUpperCase()} (Conf: {(detectionConfidence! * 100).toFixed(1)}%)</span>
                  <span>Model: {modelName?.replace('_', ' ').toUpperCase()}</span>
                  <span>Generated: {new Date(data.generatedAt).toLocaleTimeString()}</span>
                </div>
              </GlassCard>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};


