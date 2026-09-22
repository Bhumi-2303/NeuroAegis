const API_BASE_URL = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL)
  ? (import.meta.env.VITE_API_BASE_URL as string)
  : (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL)
  ? `${import.meta.env.VITE_API_URL}/api/v1`
  : 'http://127.0.0.1:8000/api/v1';

export interface PredictResponse {
  job_id: string;
  detected_dataset: string;
  confidence: number;
  matched_rules: string[];
  selected_model: string;
  validation: EdfValidationResult;
}

export interface EdfValidationResult {
  validationStatus: 'valid' | 'invalid';
  fileName: string;
  fileSizeBytes: number;
  samplingRate: number | null;
  durationSeconds: number | null;
  totalChannels: number;
  eegChannels: number;
  channelNames: string[];
  excludedChannels: string[];
  dataset: 'bonn' | 'chbmit' | 'siena' | 'unknown';
  detectionConfidence: number;
  matchedRules: string[];
  errors: string[];
}

export interface EegSeizureInterval {
  startSeconds: number;
  endSeconds: number;
  durationSeconds: number;
  source: 'dataset_annotation' | string;
}

export interface EegVisualizationChannel {
  id: string;
  name: string;
  samples: number[];
  samplingRate: number;
  unit?: 'uV' | string;
}

export interface EegReferenceSegment {
  available: boolean;
  startSeconds: number;
  endSeconds: number;
  durationSeconds: number;
  originalSampleCount: number;
  visualizationSampleCount: number;
  channels: EegVisualizationChannel[];
}

export interface EegChannelActivity {
  channelName: string;
  activityScore: number;
  baselineScore: number;
  relativeChange: number;
  metrics: Record<string, number>;
}

export interface EegVisualization {
  dataset: 'bonn' | 'chbmit' | 'siena' | 'unknown' | string;
  fileName: string;
  fileSizeBytes: number;
  patientIdentifier: string | null;
  samplingRate: number;
  durationSeconds: number;
  totalChannels: number;
  eegChannelCount: number;
  channels: EegVisualizationChannel[];
  visualizationSampleCount: number;
  originalSampleCount: number;
  timeStartSeconds: number;
  timeEndSeconds: number;
  seizures: EegSeizureInterval[];
  hasSeizureAnnotations: boolean;
  annotationStatus: 'available' | 'unavailable' | string;
  annotationSource: string | null;
  referenceAvailable: boolean;
  reference: EegReferenceSegment | null;
  channelActivity: EegChannelActivity[];
  channelActivityAvailable: boolean;
  channelActivityNote: string;
  excludedChannels: string[];
  excludedChannelDetails: Array<{ name: string; reason: string }>;
}

export class ApiRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly validation?: EdfValidationResult,
  ) {
    super(message);
    this.name = 'ApiRequestError';
  }
}

export interface JobResponse {
  job_id: string;
  status: 'Pending' | 'Processing' | 'Completed' | 'Failed' | string;
  progress: number;
  datasetName: string | null;
  detectionConfidence: number | null;
  modelName: string | null;
  result?: {
    prediction_label: 'seizure' | 'non_seizure';
    probability_seizure: number;
    confidence_band: 'low' | 'medium' | 'high';
    shap_explanation: {
      baseValue: number;
      features: Array<{
        featureName: string;
        value: number;
        contribution?: number;
        rawValue?: number;
        referenceRange?: [number, number];
      }>;
    };
    eeg_visualization?: EegVisualization;
  };
  error?: string;
}

const IN_PROGRESS_JOB_STATUSES = new Set([
  'Pending',
  'Validating',
  'Processing',
  'Validating Patient Data',
  'Feature Extraction & Signal Processing',
  'Brain Graph Construction',
  'Graph Neural Network Inference',
  'Explainable AI (SHAP)',
  'Confidence Calculation',
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isEdfValidationResult(value: unknown): value is EdfValidationResult {
  return isRecord(value)
    && (value.validationStatus === 'valid' || value.validationStatus === 'invalid')
    && typeof value.fileName === 'string'
    && typeof value.fileSizeBytes === 'number'
    && Array.isArray(value.errors);
}

function isPredictResponse(value: unknown): value is PredictResponse {
  return isRecord(value)
    && typeof value.job_id === 'string'
    && typeof value.detected_dataset === 'string'
    && typeof value.confidence === 'number'
    && Array.isArray(value.matched_rules)
    && typeof value.selected_model === 'string'
    && isEdfValidationResult(value.validation);
}

function isJobResponse(value: unknown): value is JobResponse {
  return isRecord(value)
    && typeof value.job_id === 'string'
    && typeof value.status === 'string'
    && typeof value.progress === 'number';
}

function errorDetails(payload: unknown, status: number): { message: string; validation?: EdfValidationResult } {
  const detail = isRecord(payload) ? payload.detail : undefined;
  if (typeof detail === 'string') {
    return { message: `${detail} (${status})` };
  }
  if (isRecord(detail)) {
    const message = typeof detail.message === 'string' ? detail.message : 'EEG upload failed';
    const validation = isEdfValidationResult(detail.validation) ? detail.validation : undefined;
    return { message: `${message} (${status})`, validation };
  }
  return { message: `EEG upload failed (${status})` };
}

export async function uploadEeg(
  file: File,
  options: {
    samplingRate?: number;
    channels?: string;
    patientId?: string;
    dataset?: string;
    model?: string;
    signal?: AbortSignal;
  } = {}
): Promise<PredictResponse> {
  if (!file.name.toLowerCase().endsWith('.edf')) {
    throw new ApiRequestError('Only .edf files are supported', 400);
  }

  const formData = new FormData();

  formData.append('file', file);

  if (options.samplingRate !== undefined) {
    formData.append('sampling_rate', String(options.samplingRate));
  }

  if (options.channels) {
    formData.append('channels', options.channels);
  }

  if (options.patientId) {
    formData.append('patient_id', options.patientId);
  }

  if (options.dataset) {
    formData.append('dataset', options.dataset);
  }

  if (options.model) {
    formData.append('model', options.model);
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/predict/`, {
      method: 'POST',
      body: formData,
      signal: options.signal,
    });
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err;
    throw new ApiRequestError(`Backend service unreachable: ${(err as Error)?.message || 'Connection failed'}`, 0);
  }

  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null);
    const details = errorDetails(payload, response.status);
    throw new ApiRequestError(details.message, response.status, details.validation);
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!isPredictResponse(payload)) {
    throw new ApiRequestError('Backend returned an invalid upload response (502)', 502);
  }
  return payload;
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<JobResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/jobs/${jobId}`, { signal });
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err;
    throw new ApiRequestError(`Backend service unreachable: ${(err as Error)?.message || 'Connection failed'}`, 0);
  }

  if (!response.ok) {
    const message = await response.text();
    throw new ApiRequestError(`Job request failed (${response.status}): ${message}`, response.status);
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!isJobResponse(payload)) {
    throw new ApiRequestError('Backend returned an invalid job response (502)', 502);
  }
  return payload;
}

export async function waitForJob(
  jobId: string,
  onProgress?: (job: JobResponse) => void,
  intervalMs = 500,
  timeoutMs = 5 * 60 * 1000,
  signal?: AbortSignal,
): Promise<JobResponse> {
  const deadline = Date.now() + timeoutMs;
  let consecutiveNetworkErrors = 0;
  const maxConsecutiveNetworkErrors = 3;

  while (true) {
    signal?.throwIfAborted();

    let job: JobResponse;
    try {
      job = await getJob(jobId, signal);
      consecutiveNetworkErrors = 0;
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') throw err;
      consecutiveNetworkErrors++;
      if (consecutiveNetworkErrors > maxConsecutiveNetworkErrors) {
        throw err;
      }
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
      continue;
    }

    onProgress?.(job);

    if (job.status === 'Completed') {
      return job;
    }

    if (job.status === 'Failed' || job.status.startsWith('Failed:')) {
      throw new Error(job.error ?? 'EEG prediction failed');
    }

    if (!IN_PROGRESS_JOB_STATUSES.has(job.status)) {
      throw new Error(`Backend returned an unexpected job status: ${job.status}`);
    }

    if (Date.now() >= deadline) {
      throw new Error('EEG prediction timed out while waiting for the backend');
    }

    await new Promise((resolve) => setTimeout(resolve, Math.min(intervalMs, Math.max(0, deadline - Date.now()))));
  }
}
