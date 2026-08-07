import { useState } from 'react';
import type { ModelOutput } from '@neuroaegis/model-contracts';

const MAX_FILE_SIZE = 52 * 1024 * 1024; // 52MB

export type JobStep = 'Validating' | 'Preprocessing' | 'Inference' | 'SHAP';
export const STEPS: JobStep[] = ['Validating', 'Preprocessing', 'Inference', 'SHAP'];

export function usePredictionFlow() {
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const [datasetName, setDatasetName] = useState<string | null>(null);
  const [detectionConfidence, setDetectionConfidence] = useState<number | null>(null);
  const [modelName, setModelName] = useState<string | null>(null);
  
  const [file, setFile] = useState<File | null>(null);
  const [samplingRate, setSamplingRate] = useState<string>('');
  const [channels, setChannels] = useState<string>('');
  const [validationError, setValidationError] = useState<string | null>(null);
  
  const [isUploading, setIsUploading] = useState(false);
  const [currentStep, setCurrentStep] = useState<JobStep | null>(null);
  const [data, setData] = useState<ModelOutput | null>(null);
  const [isError, setIsError] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

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

  const resetAnalysis = () => {
    setFile(null);
    setData(null);
    setIsError(false);
    setCurrentStep(null);
    setDatasetName(null);
    setDetectionConfidence(null);
    setModelName(null);
    setValidationError(null);
    setErrorMessage(null);
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
    if (samplingRate) {
      formData.append('sampling_rate', samplingRate.toString());
    }
      formData.append('channels', channels);

      console.log(`[usePredictionFlow] Upload started. API URL: ${API_URL}/api/v1/predict/`);

      const res = await fetch(`${API_URL}/api/v1/predict/`, {
        method: 'POST',
        body: formData,
      });
      
      console.log(`[usePredictionFlow] Response received. Status: ${res.status}`);
      
      if (!res.ok) {
        let errorMsg = 'Prediction request failed';
        try {
          const errorData = await res.json();
          errorMsg = errorData.detail || errorMsg;
        } catch {
          // If response isn't JSON, it might be a gateway error
        }
        throw new Error(errorMsg);
      }
      
      const { job_id } = await res.json();
      console.log(`[usePredictionFlow] Parsed response. Job ID: ${job_id}`);
      
      const MAX_POLL_ATTEMPTS = 300; // 5 minutes max
      let attempts = 0;
      let completed = false;

      while (attempts < MAX_POLL_ATTEMPTS) {
        attempts++;
        const statusRes = await fetch(`${API_URL}/api/v1/jobs/${job_id}`);
        if (!statusRes.ok) throw new Error('Failed to fetch job status');
        const statusData = await statusRes.json();
        
        // Map backend stages to UI steps
        const s = statusData.status || '';
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
               value: Math.max(statusData.result.probability_seizure, 1.0 - statusData.result.probability_seizure),
               band: statusData.result.confidence_band
            },
            explanation: statusData.result.shap_explanation,
            generatedAt: new Date().toISOString()
          };
          setDatasetName(statusData.datasetName || 'Unknown');
          setDetectionConfidence(statusData.detectionConfidence || 0);
          setModelName(statusData.modelName || 'Unknown');
          setData(responseData);
          console.log("[usePredictionFlow] Job completed successfully.");
          completed = true;
          break;
        } else if (statusData.status === 'Failed' || statusData.error) {
           throw new Error(statusData.error || 'Job failed');
        }
        
        await new Promise(r => setTimeout(r, 1000));
      }

      if (!completed) {
        throw new Error('Prediction timed out after 5 minutes.');
      }
    } catch (err: unknown) {
      console.error("[usePredictionFlow] Error reason:", err);
      setIsError(true);
      const errObj = err instanceof Error ? err : new Error(String(err));
      if (errObj.name === 'TypeError' && errObj.message.includes('fetch')) {
         setErrorMessage("Unable to connect to the NeuroAegis backend.");
      } else {
         setErrorMessage(errObj.message || 'An unexpected error occurred.');
      }
    } finally {
      setIsUploading(false);
      setCurrentStep(null);
    }
  };

  return {
    file,
    setFile,
    samplingRate,
    setSamplingRate,
    channels,
    setChannels,
    validationError,
    setValidationError,
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
  };
}
