import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  EegChannelData,
  ModelPrediction,
  PatientVitals,
  SeizureAlert,
  ModelThresholdConfig,
  ViewLifecycleState,
} from '../types/neuroaegis';
import {
  INITIAL_PATIENT,
  generateEegBuffer,
  generatePrediction,
  createInitialAlerts,
} from '../utils/mockEegGenerator';
import { Header } from '../components/common/Header';
import { SkeletonLoader } from '../components/common/SkeletonLoader';
import { ErrorState } from '../components/common/ErrorState';
import { EmptyState } from '../components/common/EmptyState';
import { EegWaveformCanvas } from '../components/eeg/EegWaveformCanvas';
import { RiskGauge } from '../components/analytics/RiskGauge';
import { ShapExplanationView } from '../components/analytics/ShapExplanationView';
import { AlertDrawer } from '../components/alerts/AlertDrawer';
import { ThresholdControlPanel } from '../components/controls/ThresholdControlPanel';
import { PatientDetailModal } from '../components/patient/PatientDetailModal';
import { RealEegMetadata } from '../components/eeg/RealEegMetadata';
import { SignalDerivedChannelActivity } from '../components/eeg/SignalDerivedChannelActivity';
import {
  ApiRequestError,
  EdfValidationResult,
  EegVisualization,
  uploadEeg,
  waitForJob,
} from '../services/api';

function formatUploadError(error: unknown): string {
  if (error instanceof ApiRequestError) {
    if (error.validation?.errors.length) return error.validation.errors.join('; ');
    return error.message;
  }
  if (error instanceof Error && error.message) return error.message;
  return 'EEG analysis failed. Select the EDF again to retry.';
}

function formatJobStatus(status: string): string {
  const labels: Record<string, string> = {
    Validating: 'Validating EDF and detecting dataset...',
    Processing: 'Processing EEG...',
    'Validating Patient Data': 'Validating patient data...',
    'Feature Extraction & Signal Processing': 'Extracting EEG features...',
    'Brain Graph Construction': 'Constructing brain graph...',
    'Graph Neural Network Inference': 'Running model inference...',
    'Explainable AI (SHAP)': 'Generating feature explanation...',
    'Confidence Calculation': 'Finalizing prediction...',
  };
  return labels[status] ?? 'Processing EEG...';
}

export const NeuroAegisDashboard: React.FC = () => {
  // Lifecycle State Discriminated Union
  const [lifecycleState, setLifecycleState] = useState<ViewLifecycleState>({
    status: 'ready',
  });

  // Clinical Telemetry States
  const [patientVitals, setPatientVitals] = useState<PatientVitals>(INITIAL_PATIENT);
  const [isStreaming, setIsStreaming] = useState<boolean>(true);
  const [audioEnabled, setAudioEnabled] = useState<boolean>(true);
  const [isManualSeizureActive, setIsManualSeizureActive] = useState<boolean>(false);
  const [isPatientModalOpen, setIsPatientModalOpen] = useState<boolean>(false);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>("No EEG file selected");
  const [uploadValidation, setUploadValidation] = useState<EdfValidationResult | null>(null);
  const [isRealAnalysis, setIsRealAnalysis] = useState<boolean>(false);
  const [realVisualization, setRealVisualization] = useState<EegVisualization | null>(null);
  const [predictionAvailable, setPredictionAvailable] = useState<boolean>(true);
  const [isUploadInFlight, setIsUploadInFlight] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Configuration State
  const [config, setConfig] = useState<ModelThresholdConfig>({
    sensitivityThreshold: 0.65,
    temporalWindowSeconds: 3,
    gainMultiplier: 1.0,
    highPassFilterHz: 0.5,
    notchFilterEnabled: true,
  });

  // Signal & Inference State
  const timeOffsetRef = useRef<number>(0);
  const [channels, setChannels] = useState<EegChannelData[]>(() =>
    generateEegBuffer(128, 0, false, config.gainMultiplier)
  );
  const [prediction, setPrediction] = useState<ModelPrediction>(() =>
    generatePrediction(false, config.sensitivityThreshold)
  );
  const [alerts, setAlerts] = useState<SeizureAlert[]>(createInitialAlerts);

  // Interval reference
  const timerRef = useRef<number | null>(null);
  const uploadInFlightRef = useRef<boolean>(false);
  const uploadAbortRef = useRef<AbortController | null>(null);

  // Live Telemetry Loop (250ms polling tick simulating live 256Hz WebSocket stream)
  useEffect(() => {
    if (lifecycleState.status !== 'ready' || !isStreaming || isRealAnalysis) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }

    timerRef.current = window.setInterval(() => {
      timeOffsetRef.current += 32;

      // Synthesize new frame buffer
      setChannels(
        generateEegBuffer(128, timeOffsetRef.current, isManualSeizureActive, config.gainMultiplier)
      );

      // Run live model inference
      const newPred = generatePrediction(isManualSeizureActive, config.sensitivityThreshold);
      setPrediction(newPred);

      // Trigger automatic alarm if threshold exceeded
      if (newPred.probabilities.seizure >= config.sensitivityThreshold) {
        setAlerts((prevAlerts) => {
          const hasRecent = prevAlerts.some((a) => !a.acknowledged && a.severity === 'critical');
          if (hasRecent) return prevAlerts;

          const newAlert: SeizureAlert = {
            id: `alt-${Date.now()}`,
            timestamp: new Date().toLocaleTimeString(),
            severity: 'critical',
            probability: newPred.probabilities.seizure,
            primaryChannel: 'T7-P7',
            durationSeconds: config.temporalWindowSeconds,
            acknowledged: false,
          };
          return [newAlert, ...prevAlerts.slice(0, 19)];
        });
      }
    }, 250);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [
    lifecycleState.status,
    isStreaming,
    isRealAnalysis,
    isManualSeizureActive,
    config.gainMultiplier,
    config.sensitivityThreshold,
    config.temporalWindowSeconds,
  ]);

  // Abort in-flight upload requests on unmount
  useEffect(() => {
    return () => {
      uploadAbortRef.current?.abort();
    };
  }, []);

  // Handlers
  const handleToggleStream = useCallback(() => {
    setIsStreaming((prev) => !prev);
  }, []);

  const handleToggleAudio = useCallback(() => {
    setAudioEnabled((prev) => !prev);
  }, []);

  const handleUploadEeg = useCallback(async (file: File) => {
    if (uploadInFlightRef.current) return;

    // Abort any previously abandoned request
    uploadAbortRef.current?.abort();
    const abortController = new AbortController();
    uploadAbortRef.current = abortController;

    uploadInFlightRef.current = true;
    setIsUploadInFlight(true);
    setIsRealAnalysis(true);
    setRealVisualization(null);
    setPredictionAvailable(false);
    setPrediction(generatePrediction(false, config.sensitivityThreshold));
    setChannels([]);
    setAlerts([]);
    setIsManualSeizureActive(false);
    setUploadedFileName(file.name);
    setUploadValidation(null);
    setUploadError(null);
    setUploadStatus('Uploading EEG...');

    if (!file.name.toLowerCase().endsWith('.edf')) {
      const message = 'Only .edf files are supported';
      setUploadError(message);
      setUploadStatus(message);
      uploadInFlightRef.current = false;
      setIsUploadInFlight(false);
      return;
    }

    try {
      const submitted = await uploadEeg(file, { signal: abortController.signal });
      setUploadValidation(submitted.validation);
      setUploadStatus('Validating EDF and detecting dataset...');
      const completed = await waitForJob(
        submitted.job_id,
        (job) => setUploadStatus(formatJobStatus(job.status)),
        500,
        5 * 60 * 1000,
        abortController.signal,
      );

      if (!completed.result) {
        throw new Error("Backend completed without a prediction result");
      }

      const visualization = completed.result.eeg_visualization;

      if (!visualization || visualization.channels.length === 0) {
        throw new Error("Backend completed without EEG visualization data");
      }

      const realChannels: EegChannelData[] = visualization.channels.map((channel) => ({
        id: channel.id,
        name: channel.name,
        voltageMicrovolts: channel.samples,
        baselineOffset: 0,
      }));

      setChannels(realChannels);
      setRealVisualization(visualization);

      const seizureProbability = completed.result.probability_seizure;
      const confidenceValue = Math.max(seizureProbability, 1 - seizureProbability);

      const modelName =
        completed.modelName === "random_forest" ||
        completed.modelName === "xgboost" ||
        completed.modelName === "lightgbm"
          ? completed.modelName
          : "lightgbm";

      const realPrediction: ModelPrediction = {
        modelName,
        label: completed.result.prediction_label,
        probabilities: {
          seizure: seizureProbability,
          non_seizure: 1 - seizureProbability,
        },
        confidence: {
          value: confidenceValue,
          band: completed.result.confidence_band,
        },
        explanation: {
          baseValue: completed.result.shap_explanation.baseValue,
          features: completed.result.shap_explanation.features.map((feature) => ({
            featureName: feature.featureName,
            value: feature.value,
            contribution: feature.value,
            rawValue: feature.rawValue,
            referenceRange: feature.referenceRange,
          })),
        },
        generatedAt: new Date().toISOString(),
      };

      setPrediction(realPrediction);
      setPredictionAvailable(true);
      setUploadError(null);
      setUploadStatus('Analysis complete');

      console.log("NeuroAegis real EEG prediction:", realPrediction);
    } catch (error) {
      // Silently ignore abort errors — they are expected on unmount or new upload
      if ((error instanceof DOMException && error.name === 'AbortError') || (error as Error)?.name === 'AbortError') return;

      const message = formatUploadError(error);
      if (error instanceof ApiRequestError && error.validation) {
        setUploadValidation(error.validation);
      }
      setRealVisualization(null);
      setPredictionAvailable(false);
      setChannels([]);
      setAlerts([]);
      setUploadError(message);
      setUploadStatus(`Analysis failed: ${message}`);
      console.error("NeuroAegis EEG analysis failed:", error);
    } finally {
      uploadInFlightRef.current = false;
      setIsUploadInFlight(false);
    }
  }, [config.gainMultiplier, config.sensitivityThreshold]);

  const handleResetToDemo = useCallback(() => {
    uploadAbortRef.current?.abort();
    setIsUploadInFlight(false);
    setIsRealAnalysis(false);
    setRealVisualization(null);
    setPredictionAvailable(true);
    setUploadedFileName(null);
    setUploadStatus("No EEG file selected");
    setUploadValidation(null);
    setUploadError(null);
    setChannels(generateEegBuffer(128, 0, false, config.gainMultiplier));
    setPrediction(generatePrediction(false, config.sensitivityThreshold));
    setAlerts(createInitialAlerts());
    setIsManualSeizureActive(false);
  }, [config.gainMultiplier, config.sensitivityThreshold]);

  const handleTriggerManualSeizure = useCallback(() => {
    setIsManualSeizureActive((prev) => !prev);
  }, []);

  const handleAcknowledgeAlert = useCallback((alertId: string) => {
    setAlerts((prev) =>
      prev.map((alt) => (alt.id === alertId ? { ...alt, acknowledged: true } : alt))
    );
  }, []);

  const handleClearAllAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  const handleExportData = useCallback(
    (format: 'json' | 'csv') => {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      let dataContent = '';
      let mimeType = '';
      let filename = '';

      if (format === 'json') {
        dataContent = JSON.stringify(
          {
            patient: isRealAnalysis
              ? { patientId: realVisualization?.patientIdentifier ?? null }
              : patientVitals,
            prediction: predictionAvailable ? prediction : null,
            alerts,
            channels: channels.map((c) => ({
              channel: c.name,
              samplesCount: c.voltageMicrovolts.length,
            })),
          },
          null,
          2
        );
        mimeType = 'application/json';
        filename = `neuroaegis_session_${timestamp}.json`;
      } else {
        const rows = [['Timestamp', 'Model', 'Label', 'SeizureProbability', 'ConfidenceBand']];
        if (predictionAvailable) {
          rows.push([
            prediction.generatedAt,
            prediction.modelName,
            prediction.label,
            prediction.probabilities.seizure.toString(),
            prediction.confidence.band,
          ]);
        }
        dataContent = rows.map((r) => r.join(',')).join('\n');
        mimeType = 'text/csv';
        filename = `neuroaegis_prediction_${timestamp}.csv`;
      }

      const blob = new Blob([dataContent], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    },
    [isRealAnalysis, patientVitals, prediction, predictionAvailable, realVisualization, alerts, channels]
  );

  const referenceChannels: EegChannelData[] = realVisualization?.reference?.channels.map((channel) => ({
    id: channel.id,
    name: channel.name,
    voltageMicrovolts: channel.samples,
    baselineOffset: 0,
  })) ?? [];



  // Render correct view state according to standard UI lifecycle
  if (lifecycleState.status === 'loading') {
    return <SkeletonLoader />;
  }

  if (lifecycleState.status === 'error') {
    return (
      <ErrorState
        message={lifecycleState.message}
        code={lifecycleState.code}
        onRetry={() => setLifecycleState({ status: 'ready' })}
      />
    );
  }

  if (lifecycleState.status === 'empty') {
    return (
      <EmptyState
        onSelectSamplePatient={() => {
          setPatientVitals(INITIAL_PATIENT);
          setLifecycleState({ status: 'ready' });
        }}
        onUploadSession={() => {
          setLifecycleState({ status: 'ready' });
        }}
      />
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* Sticky Telemetry Bar */}
      <Header
        vitals={patientVitals}
        isStreaming={isStreaming}
        audioEnabled={audioEnabled}
        onToggleStream={handleToggleStream}
        onToggleAudio={handleToggleAudio}
        onExportData={handleExportData}
        onTriggerManualSeizure={handleTriggerManualSeizure}
        isManualSeizureActive={isManualSeizureActive}
        onOpenPatientModal={() => setIsPatientModalOpen(true)}
        isRealAnalysis={isRealAnalysis}
        patientIdentifier={realVisualization?.patientIdentifier}
      />

      {/* Main Clinical Monitoring Grid */}
      <main className="flex-1 p-4 lg:p-6 max-w-7xl w-full mx-auto space-y-6">
        {/* Quick Simulation States Bar for testing and evaluation */}
        <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/60 border border-slate-800/80 px-4 py-2.5 rounded-xl text-xs">
          <span className="text-slate-400">
            Lifecycle State Inspector:
          </span>
          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={() => {
                setLifecycleState({ status: 'loading' });
                setTimeout(() => setLifecycleState({ status: 'ready' }), 1200);
              }}
              className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition text-[11px]"
            >
              Simulate Loading
            </button>
            <button
              type="button"
              onClick={() =>
                setLifecycleState({
                  status: 'error',
                  message: 'High-frequency telemetry drop in EEG acquisition node.',
                  code: 'ERR_DSP_BUFFER_OVERFLOW_0x2A',
                  retryCount: 1,
                })
              }
              className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition text-[11px]"
            >
              Simulate Error
            </button>
            <button
              type="button"
              onClick={() => setLifecycleState({ status: 'empty' })}
              className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition text-[11px]"
            >
              Simulate Empty
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/60 border border-slate-800/80 px-4 py-3 rounded-xl">
          <div>
            <div className={`text-sm font-medium ${uploadError ? 'text-rose-300' : isRealAnalysis ? 'text-emerald-300' : 'text-slate-200'}`}>
              {uploadError ? 'EEG ANALYSIS ERROR' : isRealAnalysis ? 'REAL EEG ANALYSIS' : 'DEMO / SIMULATION MODE'}
            </div>
            <div className="text-xs text-slate-500" aria-live="polite">
              {uploadedFileName ? uploadedFileName + " • " + uploadStatus : uploadStatus}
            </div>
            {uploadError && (
              <div role="alert" className="mt-2 text-xs text-rose-300">
                {uploadError} Select another EDF or retry the upload.
              </div>
            )}
            {uploadValidation && (
              <div className="mt-2 text-[11px] text-slate-400">
                Dataset: {uploadValidation.dataset} · Size: {(uploadValidation.fileSizeBytes / 1_000_000).toFixed(2)} MB ·
                Sampling: {uploadValidation.samplingRate ?? "-"} Hz · Channels: {uploadValidation.eegChannels}/{uploadValidation.totalChannels} ·
                Duration: {uploadValidation.durationSeconds == null ? "-" : `${uploadValidation.durationSeconds.toFixed(1)} s`}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            {isRealAnalysis && (
              <button
                type="button"
                onClick={handleResetToDemo}
                disabled={isUploadInFlight}
                className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Reset to Demo
              </button>
            )}
            <label className={`px-3 py-2 rounded-lg bg-indigo-600 text-white text-xs font-medium transition ${isUploadInFlight ? 'cursor-not-allowed opacity-60' : 'cursor-pointer hover:bg-indigo-500'}`}>
              {isUploadInFlight ? 'Analyzing EEG...' : 'Upload EEG'}
              <input
                type="file"
                accept=".edf"
                className="hidden"
                disabled={isUploadInFlight}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void handleUploadEeg(file);
                  event.target.value = "";
                }}
              />
            </label>
          </div>
        </div>

        {isRealAnalysis && realVisualization && (
          <RealEegMetadata visualization={realVisualization} />
        )}

        {/* Real-time Waveform Canvas + Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: 16-Channel Waveform Stream (8 cols) */}
          <div className="lg:col-span-8 space-y-6 flex flex-col">
            <EegWaveformCanvas
              channels={channels}
              isSeizureActive={!isRealAnalysis && (
                isManualSeizureActive ||
                prediction.probabilities.seizure >= config.sensitivityThreshold
              )}
              gain={config.gainMultiplier}
              samplingRate={realVisualization?.samplingRate}
              durationSeconds={realVisualization?.durationSeconds}
              timeStartSeconds={realVisualization?.timeStartSeconds}
              timeEndSeconds={realVisualization?.timeEndSeconds}
              seizures={realVisualization?.seizures}
              title={isRealAnalysis ? 'PATIENT EEG - SEIZURE ANALYSIS' : 'DEMO EEG - SIMULATION'}
              emptyMessage={isRealAnalysis ? uploadStatus : 'No waveform available'}
            />

            {isRealAnalysis && realVisualization && (
              <section className="space-y-3">
                <div>
                  <h2 className="text-xs uppercase font-bold tracking-wider text-slate-300">NORMAL / REFERENCE EEG</h2>
                  <p className="text-[11px] text-slate-500 mt-1">A genuine non-seizure segment selected from this recording using dataset annotations.</p>
                </div>
                {realVisualization.referenceAvailable && realVisualization.reference ? (
                  <EegWaveformCanvas
                    channels={referenceChannels}
                    gain={config.gainMultiplier}
                    samplingRate={realVisualization.samplingRate}
                    durationSeconds={realVisualization.reference.durationSeconds}
                    title="NORMAL / REFERENCE EEG"
                    emptyMessage="No reference waveform available"
                  />
                ) : (
                  <p className="text-xs text-slate-500 py-4">No reference EEG segment is available for this recording.</p>
                )}
              </section>
            )}

            {/* Inference & DSP Tuning Panel */}
            <ThresholdControlPanel config={config} onChange={setConfig} />
          </div>

          {/* Right Column: AI Analytics & Alarm Queues (4 cols) */}
          <div className="lg:col-span-4 space-y-6 flex flex-col">
            {/* Real-time Probability Gauge */}
            {predictionAvailable ? (
              <RiskGauge
                prediction={prediction}
                sensitivityThreshold={config.sensitivityThreshold}
              />
            ) : (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 text-sm text-slate-500 shadow-lg">
                Model prediction unavailable until the EDF analysis completes.
              </div>
            )}

            {/* SHAP Feature Contribution (XAI) */}
            <ShapExplanationView explanation={predictionAvailable ? prediction.explanation : null} />

            {isRealAnalysis && realVisualization && (
              <SignalDerivedChannelActivity
                activity={realVisualization.channelActivity}
                available={realVisualization.channelActivityAvailable}
                note={realVisualization.channelActivityNote}
              />
            )}

            {/* Clinical Alarm List */}
            {!isRealAnalysis && (
              <AlertDrawer
                alerts={alerts}
                onAcknowledgeAlert={handleAcknowledgeAlert}
                onClearAll={handleClearAllAlerts}
              />
            )}
            {isRealAnalysis && (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-xs text-slate-500 shadow-lg">
                Dataset seizure annotations are shown on the patient EEG. The model prediction remains a window-level result.
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Patient Profile Modal */}
      {!isRealAnalysis && (
        <PatientDetailModal
          isOpen={isPatientModalOpen}
          vitals={patientVitals}
          onClose={() => setIsPatientModalOpen(false)}
        />
      )}
    </div>
  );
};

export default NeuroAegisDashboard;
