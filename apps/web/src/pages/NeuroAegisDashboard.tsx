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

  // Live Telemetry Loop (250ms polling tick simulating live 256Hz WebSocket stream)
  useEffect(() => {
    if (lifecycleState.status !== 'ready' || !isStreaming) {
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
    isManualSeizureActive,
    config.gainMultiplier,
    config.sensitivityThreshold,
    config.temporalWindowSeconds,
  ]);

  // Handlers
  const handleToggleStream = useCallback(() => {
    setIsStreaming((prev) => !prev);
  }, []);

  const handleToggleAudio = useCallback(() => {
    setAudioEnabled((prev) => !prev);
  }, []);

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
            patient: patientVitals,
            prediction,
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
        const rows = [
          ['Timestamp', 'Model', 'Label', 'SeizureProbability', 'ConfidenceBand'],
          [
            prediction.generatedAt,
            prediction.modelName,
            prediction.label,
            prediction.probabilities.seizure.toString(),
            prediction.confidence.band,
          ],
        ];
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
    [patientVitals, prediction, alerts, channels]
  );

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

        {/* Real-time Waveform Canvas + Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: 16-Channel Waveform Stream (8 cols) */}
          <div className="lg:col-span-8 space-y-6 flex flex-col">
            <EegWaveformCanvas
              channels={channels}
              isSeizureActive={
                isManualSeizureActive ||
                prediction.probabilities.seizure >= config.sensitivityThreshold
              }
              gain={config.gainMultiplier}
            />

            {/* Inference & DSP Tuning Panel */}
            <ThresholdControlPanel config={config} onChange={setConfig} />
          </div>

          {/* Right Column: AI Analytics & Alarm Queues (4 cols) */}
          <div className="lg:col-span-4 space-y-6 flex flex-col">
            {/* Real-time Probability Gauge */}
            <RiskGauge
              prediction={prediction}
              sensitivityThreshold={config.sensitivityThreshold}
            />

            {/* SHAP Feature Contribution (XAI) */}
            <ShapExplanationView explanation={prediction.explanation} />

            {/* Clinical Alarm List */}
            <AlertDrawer
              alerts={alerts}
              onAcknowledgeAlert={handleAcknowledgeAlert}
              onClearAll={handleClearAllAlerts}
            />
          </div>
        </div>
      </main>

      {/* Patient Profile Modal */}
      <PatientDetailModal
        isOpen={isPatientModalOpen}
        vitals={patientVitals}
        onClose={() => setIsPatientModalOpen(false)}
      />
    </div>
  );
};

export default NeuroAegisDashboard;
