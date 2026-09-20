const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

export interface PredictResponse {
  job_id: string;
  detected_dataset: string;
  confidence: number;
  matched_rules: string[];
  selected_model: string;
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
        rawValue?: number;
        referenceRange?: [number, number];
      }>;
    };
    eeg_visualization?: {
      samplingRate: number;
      originalSampleCount: number;
      visualizationSampleCount: number;
      channels: Array<{
        id: string;
        name: string;
        samples: number[];
      }>;
    };
  };
  error?: string;
}

export async function uploadEeg(
  file: File,
  options: {
    samplingRate?: number;
    channels?: string;
    patientId?: string;
    dataset?: string;
    model?: string;
  } = {}
): Promise<PredictResponse> {
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

  const response = await fetch(`${API_BASE_URL}/predict/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`EEG upload failed (${response.status}): ${message}`);
  }

  return response.json();
}

export async function getJob(jobId: string): Promise<JobResponse> {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`);

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Job request failed (${response.status}): ${message}`);
  }

  return response.json();
}

export async function waitForJob(
  jobId: string,
  onProgress?: (job: JobResponse) => void,
  intervalMs = 500
): Promise<JobResponse> {
  while (true) {
    const job = await getJob(jobId);

    onProgress?.(job);

    if (job.status === 'Completed') {
      return job;
    }

    if (job.status === 'Failed') {
      throw new Error(job.error ?? 'EEG prediction failed');
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
