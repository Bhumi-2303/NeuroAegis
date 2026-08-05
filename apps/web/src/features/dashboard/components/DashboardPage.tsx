import React, { useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Upload, FileText } from 'lucide-react';
import { ErrorState } from '../../../shared/components';
import { pageTransition, staggerChildren, fadeIn } from '../../../shared/lib/motion-presets';
import { usePredictionFlow } from '../../../shared/hooks';

// New Widgets
import { WidgetCard } from './widgets/WidgetCard';
import { DatasetDetectionCard } from './widgets/DatasetDetectionCard';
import { SignalQualityCard } from './widgets/SignalQualityCard';
import { ModelInfoCard } from './widgets/ModelInfoCard';
import { PredictionSummaryCard } from './widgets/PredictionSummaryCard';
import { EEGViewer } from './widgets/EEGViewer';
import { RawFilteredComparison } from './widgets/RawFilteredComparison';
import { FeatureSummaryCard } from './widgets/FeatureSummaryCard';
import { TimeFrequencyAnalysis } from './widgets/TimeFrequencyAnalysis';
import { EnhancedShapPanel } from './widgets/EnhancedShapPanel';
import { GlobalFeatureImportance } from './widgets/GlobalFeatureImportance';
import { ClinicalSummaryReport } from './widgets/ClinicalSummaryReport';
import { ExportPanel } from './widgets/ExportPanel';
import { LoadingPipeline } from './widgets/LoadingPipeline';

const MOCK_STATIC_WAVEFORM = Array.from({ length: 400 }, (_, i) => Math.sin(i * 0.1) * 50 + (Math.random() * 20 - 10));

export function DashboardPage(): React.JSX.Element {
  const {
    file,
    samplingRate,
    setSamplingRate,
    channels,
    setChannels,
    validationError,
    isUploading,
    data,
    isError,
    errorMessage,
    datasetName,
    detectionConfidence,
    handleFileChange,
    resetAnalysis,
    handlePredict
  } = usePredictionFlow();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [rawSignal, setRawSignal] = React.useState<number[]>([]);

  const [isDragging, setIsDragging] = React.useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles && droppedFiles.length > 0) {
      const syntheticEvent = {
        target: { files: droppedFiles }
      } as unknown as React.ChangeEvent<HTMLInputElement>;
      handleFileChange(syntheticEvent);
    }
  };

  React.useEffect(() => {
    if (file && data) {
      if (file.name.toLowerCase().endsWith('.edf')) {
        // Binary EDF files cannot be parsed as plain text
        setRawSignal(MOCK_STATIC_WAVEFORM);
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        if (content) {
          const lines = content.split('\n');

          const vals: number[] = [];
          for (const line of lines) {
             const val = parseFloat(line.trim());
             if (!isNaN(val)) vals.push(val);
             if (vals.length >= 1000) break;
          }
          setRawSignal(vals.length > 0 ? vals : MOCK_STATIC_WAVEFORM);
        }
      };
      reader.readAsText(file);
    } else {
      setRawSignal([]);
    }
  }, [file, data]);

  return (
    <motion.div {...pageTransition} className="flex flex-col gap-6 pb-20">
      {/* Page Title */}
      <div className="flex items-center gap-3 mb-2">
        <div className="p-2 bg-[var(--accent-primary)]/10 text-[var(--accent-primary)] rounded-xl">
          <Brain size={24} strokeWidth={2} />
        </div>
        <div>
          <h1 id="dashboard-title" className="text-xl font-bold text-[var(--text-primary)] m-0 tracking-tight">
            NeuroAegis Analysis Interface
          </h1>
          <p className="text-[13px] text-[var(--text-secondary)] m-0 mt-0.5 font-medium">
            Biomedical research software for EEG pattern recognition.
          </p>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {/* Idle State: File Upload Form */}
        {!isUploading && !isError && !data && (
          <motion.div key="idle" {...fadeIn} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            <WidgetCard title="Signal Input Configuration" className="lg:col-span-1 h-fit shadow-sm">
              <div className="space-y-6">
                <div>
                  <label className="block text-xs font-semibold text-[var(--text-secondary)] mb-2 uppercase tracking-wider">Source File</label>
                  <div 
                    className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
                      isDragging 
                        ? 'border-[var(--accent-primary)] bg-[var(--accent-primary)]/20 scale-[1.01]' 
                        : file 
                        ? 'border-[var(--accent-primary)] bg-[var(--accent-primary)]/10' 
                        : 'border-[var(--bg-4)] hover:border-[var(--accent-primary)] hover:bg-[var(--bg-3)]'
                    }`}
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
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
                        <FileText className="w-8 h-8 text-[var(--accent-primary)]" />
                        <span className="text-sm font-semibold text-[var(--text-primary)]">{file.name}</span>
                        <span className="text-xs text-[var(--text-secondary)] font-mono">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-3">
                        <Upload className="w-8 h-8 text-[var(--text-muted)]" />
                        <span className="text-sm font-medium text-[var(--text-secondary)]">Drop EEG file here</span>
                        <span className="text-xs text-[var(--text-muted)] font-mono">.edf, .csv, .txt (Max 52MB)</span>
                      </div>
                    )}
                  </div>
                  {validationError && (
                    <p className="mt-2 text-xs text-[var(--state-danger)] font-medium">{validationError}</p>
                  )}
                </div>

                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="block text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider">Sampling Rate</label>
                    <span className="text-[10px] text-[var(--text-muted)] font-mono">Hz</span>
                  </div>
                  <input 
                    type="number" 
                    placeholder="Auto-detect"
                    value={samplingRate}
                    onChange={e => setSamplingRate(e.target.value)}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-4)] rounded-lg px-3 py-2.5 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-secondary)] focus:ring-1 focus:ring-[var(--accent-secondary)] shadow-sm transition-shadow"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-[var(--text-secondary)] mb-1 uppercase tracking-wider">Channels</label>
                  <input 
                    type="text" 
                    value={channels}
                    onChange={e => setChannels(e.target.value)}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-4)] rounded-lg px-3 py-2.5 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-secondary)] focus:ring-1 focus:ring-[var(--accent-secondary)] shadow-sm transition-shadow"
                    placeholder="e.g. EEG-Fpz-Cz"
                  />
                </div>

                <button 
                  onClick={handlePredict}
                  disabled={isUploading || !file}
                  className="w-full py-3 rounded-lg bg-[var(--accent-primary)] text-white font-semibold text-sm flex items-center justify-center gap-2 hover:bg-[var(--accent-secondary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md mt-4"
                >
                  <Brain className="w-4 h-4" />
                  Initialize Analysis
                </button>
              </div>
            </WidgetCard>

            <div className="lg:col-span-2 flex flex-col justify-center items-center p-8 bg-[var(--bg-3)] rounded-2xl border border-[var(--bg-4)] border-dashed">
               <div className="w-16 h-16 bg-[var(--bg-2)] rounded-2xl shadow-sm border border-[var(--bg-4)] flex items-center justify-center mb-6">
                 <Brain size={32} className="text-[var(--text-muted)]" />
               </div>
               <h2 className="text-xl font-bold text-[var(--text-primary)] mb-2 tracking-tight">Research Framework Ready</h2>
               <p className="text-sm text-[var(--text-secondary)] text-center max-w-sm leading-relaxed">
                 Upload multi-channel or single-channel EEG signals to begin automated feature extraction and inference.
               </p>
            </div>
            
          </motion.div>
        )}

        {/* Loading State: Pipeline */}
        {isUploading && (
          <motion.div key="loading" {...fadeIn}>
            <LoadingPipeline />
          </motion.div>
        )}

        {/* Error State */}
        {isError && !isUploading && (
          <motion.div key="error" {...fadeIn} className="max-w-2xl mx-auto mt-20">
            <ErrorState 
              title="Pipeline Execution Failed" 
              message={errorMessage || "An error occurred during feature extraction or inference."} 
              onRetry={resetAnalysis} 
              retryLabel="Reset Environment" 
            />
          </motion.div>
        )}

        {/* Results State */}
        {!isUploading && !isError && data && (
          <motion.div key="results" {...staggerChildren} className="flex flex-col gap-6 w-full max-w-7xl mx-auto">
            
            {/* Header / Upload again */}
            <div className="flex justify-between items-center bg-[var(--bg-2)] p-4 rounded-xl border border-[var(--bg-4)] shadow-sm">
              <div className="flex items-center gap-4">
                <div className="p-2 bg-[var(--state-success)]/10 text-[var(--state-success)] rounded-lg">
                  <FileText size={20} />
                </div>
                <div>
                  <span className="block text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider mb-0.5">Active Workspace</span>
                  <span className="block text-sm font-semibold text-[var(--text-primary)]">{file?.name}</span>
                </div>
              </div>
              <button 
                onClick={resetAnalysis}
                className="px-4 py-2 rounded-lg bg-[var(--bg-2)] text-[var(--text-secondary)] font-semibold text-xs flex items-center gap-2 hover:bg-[var(--bg-3)] transition-colors border border-[var(--bg-4)]"
              >
                <Upload size={14} />
                New Analysis
              </button>
            </div>

            {/* Row 1: Primary Prediction Summary */}
            <PredictionSummaryCard 
              probability={data.prediction.probabilities.seizure}
              label={data.prediction.label}
              datasetName={datasetName || undefined}
              modelName={data.modelName}
            />

            {/* Row 2: Signal Visualization */}
            <EEGViewer data={rawSignal.length > 0 ? rawSignal : MOCK_STATIC_WAVEFORM} />

            {/* Row 3: Explainability */}
            <EnhancedShapPanel data={data} />

            {/* Row 4: 2-column grid for Dataset and Model Info */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <DatasetDetectionCard 
                datasetName={datasetName || undefined} 
                confidence={detectionConfidence || undefined} 
                samplingRate={samplingRate} 
                channels={channels} 
              />
              <ModelInfoCard modelName={data.modelName} datasetName={datasetName || undefined} />
            </div>

            {/* Row 5: Features Summary */}
            <FeatureSummaryCard />

            {/* Row 8: Global Importance & System Performance (split) */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
              <div className="xl:col-span-2 h-full">
                <GlobalFeatureImportance />
              </div>
              <div className="xl:col-span-1 flex flex-col gap-6">
                <ExportPanel data={data} datasetName={datasetName} />
              </div>
            </div>

            {/* Row 9: Clinical Report */}
            <ClinicalSummaryReport data={data} file={file} />
            
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
