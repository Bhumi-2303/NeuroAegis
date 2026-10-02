export const API_BASE_URL = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL)
  ? (import.meta.env.VITE_API_BASE_URL as string)
  : (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL)
  ? `${import.meta.env.VITE_API_URL}/api/v1`
  : 'http://127.0.0.1:8000/api/v1';

export interface AuthUser {
  id: string;
  username: string;
  tenant_id: string;
  role: 'admin' | 'clinician' | 'researcher';
  is_active: boolean;
}

export interface LoginResponse {
  csrf_token: string;
  user: AuthUser;
}

export interface PredictResponse {
  job_id: string;
  patient_id?: string;
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
  readonly status: number;
  readonly validation?: EdfValidationResult;
  readonly requestId?: string;

  constructor(
    message: string,
    status: number,
    validation?: EdfValidationResult,
    requestId?: string,
  ) {
    super(message);
    this.name = 'ApiRequestError';
    this.status = status;
    this.validation = validation;
    this.requestId = requestId;
  }
}

export function generateRequestId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return 'req-' + Math.random().toString(36).substring(2, 15) + '-' + Date.now().toString(36);
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

// ==============================================================================
// CSRF & Session Management
// ==============================================================================

let inMemoryCsrfToken: string | null = null;

export function setCsrfToken(token: string | null): void {
  inMemoryCsrfToken = token;
}

export function getCsrfTokenFromCookie(): string | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(/(?:^|;\s*)neuroaegis_csrf_token=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}

export function getCsrfToken(): string | null {
  return getCsrfTokenFromCookie() || inMemoryCsrfToken;
}

type SessionExpiredHandler = () => void;
const sessionExpiredListeners: Set<SessionExpiredHandler> = new Set();

export function onSessionExpired(handler: SessionExpiredHandler): () => void {
  sessionExpiredListeners.add(handler);
  return () => {
    sessionExpiredListeners.delete(handler);
  };
}

function notifySessionExpired(): void {
  for (const listener of sessionExpiredListeners) {
    try {
      listener();
    } catch {}
  }
}

let refreshPromise: Promise<boolean> | null = null;

export async function refreshAuthSession(): Promise<boolean> {
  if (refreshPromise) return refreshPromise;

  refreshPromise = (async () => {
    try {
      const csrf = getCsrfToken();
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'X-Request-ID': generateRequestId(),
      };
      if (csrf) {
        headers['X-CSRF-Token'] = csrf;
      }

      const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
        method: 'POST',
        headers,
        credentials: 'include',
      });

      if (response.ok) {
        const payload = await response.json().catch(() => null);
        if (payload?.csrf_token) {
          setCsrfToken(payload.csrf_token);
        }
        return true;
      }
      return false;
    } catch {
      return false;
    } finally {
      refreshPromise = null;
    }
  })();

  return refreshPromise;
}

// ==============================================================================
// Centralized API Client (apiFetch)
// ==============================================================================

export async function apiFetch(
  endpoint: string,
  options: RequestInit & { _retry?: boolean; requestId?: string } = {}
): Promise<Response> {
  const url = endpoint.startsWith('http')
    ? endpoint
    : `${API_BASE_URL}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;
  const method = (options.method || 'GET').toUpperCase();
  const isStateChanging = ['POST', 'PUT', 'PATCH', 'DELETE'].includes(method);

  const headers = new Headers(options.headers || {});

  // Determine or assign correlation ID
  const effectiveRequestId =
    options.requestId ||
    headers.get('X-Request-ID') ||
    headers.get('x-request-id') ||
    generateRequestId();

  if (!headers.has('X-Request-ID') && !headers.has('x-request-id')) {
    headers.set('X-Request-ID', effectiveRequestId);
  }

  // Add CSRF token for state-changing requests when not the initial login endpoint
  if (isStateChanging && !url.endsWith('/auth/login')) {
    const csrfToken = getCsrfToken();
    if (csrfToken && !headers.has('X-CSRF-Token') && !headers.has('x-csrf-token')) {
      headers.set('X-CSRF-Token', csrfToken);
    }
  }

  const requestOptions: RequestInit = {
    ...options,
    headers,
    credentials: 'include',
  };

  let response: Response;
  try {
    response = await fetch(url, requestOptions);
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err;
    throw new ApiRequestError(
      `Backend service unreachable: ${(err as Error)?.message || 'Connection failed'}`,
      0,
      undefined,
      effectiveRequestId,
    );
  }

  // Centrally handle 401 Unauthorized for protected endpoints
  if (response.status === 401 && !url.includes('/auth/login') && !url.includes('/auth/refresh')) {
    if (!options._retry && Boolean(getCsrfToken())) {
      const refreshed = await refreshAuthSession();
      if (refreshed) {
        // Retry original request once, preserving the correlation ID
        return apiFetch(endpoint, {
          ...options,
          headers,
          requestId: effectiveRequestId,
          _retry: true,
        });
      }
    }
    notifySessionExpired();
    const serverRequestId = response.headers.get('x-request-id') || effectiveRequestId;
    throw new ApiRequestError('Session expired. Please log in again.', 401, undefined, serverRequestId);
  }

  return response;
}

// ==============================================================================
// Authentication API Endpoints
// ==============================================================================

export async function loginUser(username: string, password: string): Promise<AuthUser> {
  const response = await apiFetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = isRecord(payload) && typeof payload.detail === 'string'
      ? payload.detail
      : 'Authentication failed';
    const serverRequestId = response.headers.get('x-request-id') || undefined;
    throw new ApiRequestError(detail, response.status, undefined, serverRequestId);
  }

  const payload = await response.json().catch(() => null);
  if (isRecord(payload) && typeof payload.csrf_token === 'string') {
    setCsrfToken(payload.csrf_token);
  }
  if (isRecord(payload) && isRecord(payload.user)) {
    return payload.user as unknown as AuthUser;
  }
  return getMe();
}

export async function getMe(): Promise<AuthUser> {
  const response = await apiFetch('/auth/me');
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = isRecord(payload) && typeof payload.detail === 'string'
      ? payload.detail
      : 'Failed to retrieve current user session';
    const serverRequestId = response.headers.get('x-request-id') || undefined;
    throw new ApiRequestError(detail, response.status, undefined, serverRequestId);
  }
  const payload = await response.json().catch(() => null);
  if (!isRecord(payload) || typeof payload.id !== 'string') {
    const serverRequestId = response.headers.get('x-request-id') || undefined;
    throw new ApiRequestError('Invalid user profile response', 502, undefined, serverRequestId);
  }
  return payload as unknown as AuthUser;
}

export async function logoutUser(): Promise<void> {
  try {
    await apiFetch('/auth/logout', { method: 'POST' });
  } finally {
    setCsrfToken(null);
    notifySessionExpired();
  }
}

export async function logoutAllSessions(): Promise<void> {
  try {
    await apiFetch('/auth/logout-all', { method: 'POST' });
  } finally {
    setCsrfToken(null);
    notifySessionExpired();
  }
}

// ==============================================================================
// Clinical Pipeline (EDF Upload & Job Telemetry)
// ==============================================================================

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
    response = await apiFetch('/predict/', {
      method: 'POST',
      body: formData,
      signal: options.signal,
    });
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err;
    if (err instanceof ApiRequestError) throw err;
    throw new ApiRequestError(`Backend service unreachable: ${(err as Error)?.message || 'Connection failed'}`, 0);
  }

  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null);
    const details = errorDetails(payload, response.status);
    const serverRequestId = response.headers.get('x-request-id') || undefined;
    throw new ApiRequestError(details.message, response.status, details.validation, serverRequestId);
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!isPredictResponse(payload)) {
    const serverRequestId = response.headers.get('x-request-id') || undefined;
    throw new ApiRequestError('Backend returned an invalid upload response (502)', 502, undefined, serverRequestId);
  }
  return payload;
}

export async function uploadEegV2(
  file: File,
  patientData: {
    name: string;
    age: number;
    gender: string;
    weight: number;
    height: number;
    medicalHistory?: Record<string, unknown>;
    vitalSigns?: Record<string, unknown>;
  },
  options: {
    patientId?: string;
    samplingRate?: number;
    channels?: string;
    signal?: AbortSignal;
  } = {}
): Promise<PredictResponse> {
  if (!file.name.toLowerCase().endsWith('.edf')) {
    throw new ApiRequestError('Only .edf files are supported', 400);
  }

  const formData = new FormData();
  formData.append('file', file);
  formData.append('name', patientData.name);
  formData.append('age', String(patientData.age));
  formData.append('gender', patientData.gender);
  formData.append('weight', String(patientData.weight));
  formData.append('height', String(patientData.height));
  formData.append('medical_history', JSON.stringify(patientData.medicalHistory || {}));
  formData.append('vital_signs', JSON.stringify(patientData.vitalSigns || {}));

  if (options.patientId) {
    formData.append('patient_id', options.patientId);
  }
  if (options.samplingRate !== undefined) {
    formData.append('sampling_rate', String(options.samplingRate));
  }
  if (options.channels) {
    formData.append('channels', options.channels);
  }

  const v2BaseUrl = API_BASE_URL.replace(/\/api\/v1\/?$/, '/api/v2');
  let response: Response;
  try {
    response = await apiFetch(`${v2BaseUrl}/predict`, {
      method: 'POST',
      body: formData,
      signal: options.signal,
    });
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err;
    if (err instanceof ApiRequestError) throw err;
    throw new ApiRequestError(`Backend service unreachable: ${(err as Error)?.message || 'Connection failed'}`, 0);
  }

  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null);
    const details = errorDetails(payload, response.status);
    const serverRequestId = response.headers.get('x-request-id') || undefined;
    throw new ApiRequestError(details.message, response.status, details.validation, serverRequestId);
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!isPredictResponse(payload)) {
    const serverRequestId = response.headers.get('x-request-id') || undefined;
    throw new ApiRequestError('Backend returned an invalid upload response (502)', 502, undefined, serverRequestId);
  }
  return payload;
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<JobResponse> {
  let response: Response;
  try {
    response = await apiFetch(`/jobs/${jobId}`, { signal });
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err;
    if (err instanceof ApiRequestError) throw err;
    throw new ApiRequestError(`Backend service unreachable: ${(err as Error)?.message || 'Connection failed'}`, 0);
  }

  if (!response.ok) {
    const message = await response.text();
    const serverRequestId = response.headers.get('x-request-id') || undefined;
    throw new ApiRequestError(`Job request failed (${response.status}): ${message}`, response.status, undefined, serverRequestId);
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!isJobResponse(payload)) {
    const serverRequestId = response.headers.get('x-request-id') || undefined;
    throw new ApiRequestError('Backend returned an invalid job response (502)', 502, undefined, serverRequestId);
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
