import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DashboardPage } from './DashboardPage';
import { usePredictionFlow } from '../../../shared/hooks/usePredictionFlow';
import React from 'react';

// Mock the components used in DashboardPage
vi.mock('../../../shared/components', () => ({
  GlassCard: ({ children, title }: { children: React.ReactNode, title: string }) => (
    <div data-testid="glass-card">
      {title && <h2>{title}</h2>}
      {children}
    </div>
  ),
  EmptyState: () => <div data-testid="empty-state" />,
  ErrorState: ({ title, message }: { title: string, message: string }) => (
    <div data-testid="error-state">
      <h2>{title}</h2>
      <p>{message}</p>
    </div>
  ),
  ConfidenceGauge: ({ label }: { label: string }) => <div data-testid="confidence-gauge">{label}</div>,
  StatusBadge: ({ label }: { label: string }) => <div data-testid="status-badge">{label}</div>,
  Scene: () => <div data-testid="scene" />,
  WaveformSparkline: () => <div data-testid="waveform-sparkline" />
}));

// Mock the usePredictionFlow hook
vi.mock('../../../shared/hooks/usePredictionFlow', () => ({
  usePredictionFlow: vi.fn(),
  STEPS: ['Step 1', 'Step 2']
}));

describe('DashboardPage Rendering States', () => {
  const mockHandlePredict = vi.fn();
  const mockHandleFileChange = vi.fn();
  const mockResetAnalysis = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders Idle state correctly when no file is uploading, no error, and no data', () => {
    (usePredictionFlow as any).mockReturnValue({
      file: null,
      setFile: vi.fn(),
      samplingRate: 256,
      setSamplingRate: vi.fn(),
      channels: 'EEG-Fpz-Cz',
      setChannels: vi.fn(),
      validationError: null,
      isUploading: false,
      currentStep: null,
      data: null,
      isError: false,
      handleFileChange: mockHandleFileChange,
      resetAnalysis: mockResetAnalysis,
      handlePredict: mockHandlePredict,
    });

    render(<DashboardPage />);

    expect(screen.getByText('NeuroAegis Command Center')).toBeInTheDocument();
    expect(screen.getByText('EEG Upload')).toBeInTheDocument();
    expect(screen.getByTestId('scene')).toBeInTheDocument();
    expect(screen.queryByTestId('error-state')).not.toBeInTheDocument();
    expect(screen.queryByTestId('confidence-gauge')).not.toBeInTheDocument();
  });

  it('renders Loading state correctly when isUploading is true', () => {
    (usePredictionFlow as any).mockReturnValue({
      file: new File([''], 'test.csv'),
      isUploading: true,
      currentStep: 'Step 1',
      data: null,
      isError: false,
    });

    render(<DashboardPage />);

    expect(screen.getByText('Analysis in Progress')).toBeInTheDocument();
    expect(screen.getByText('Step 1')).toBeInTheDocument();
    expect(screen.queryByText('EEG Upload')).not.toBeInTheDocument();
    expect(screen.queryByTestId('scene')).not.toBeInTheDocument();
    expect(screen.queryByTestId('error-state')).not.toBeInTheDocument();
  });

  it('renders Error state correctly when isError is true', () => {
    (usePredictionFlow as any).mockReturnValue({
      file: null,
      isUploading: false,
      currentStep: null,
      data: null,
      isError: true,
      resetAnalysis: mockResetAnalysis,
    });

    render(<DashboardPage />);

    expect(screen.getByTestId('error-state')).toBeInTheDocument();
    expect(screen.getByText('Prediction Failed')).toBeInTheDocument();
    expect(screen.queryByText('EEG Upload')).not.toBeInTheDocument();
    expect(screen.queryByTestId('scene')).not.toBeInTheDocument();
  });

  it('renders Results state correctly when data is available', () => {
    (usePredictionFlow as any).mockReturnValue({
      file: new File([''], 'test.csv'),
      samplingRate: 256,
      channels: 'EEG-Fpz-Cz',
      isUploading: false,
      currentStep: null,
      data: {
        confidence: { value: 0.95, band: 'gamma' },
        prediction: {
          label: 'seizure',
          probabilities: { seizure: 0.95, non_seizure: 0.05 }
        },
        generatedAt: new Date().toISOString()
      },
      datasetName: 'chbmit',
      detectionConfidence: 0.98,
      modelName: 'chbmit_model',
      isError: false,
      resetAnalysis: mockResetAnalysis,
    });

    render(<DashboardPage />);

    expect(screen.getByText('Analysis Complete')).toBeInTheDocument();
    expect(screen.getByTestId('confidence-gauge')).toBeInTheDocument();
    expect(screen.getByText('Probability Breakdown')).toBeInTheDocument();
    expect(screen.getByTestId('status-badge')).toBeInTheDocument();
    expect(screen.queryByText('EEG Upload')).not.toBeInTheDocument();
    expect(screen.queryByTestId('scene')).not.toBeInTheDocument();
    expect(screen.queryByTestId('error-state')).not.toBeInTheDocument();
  });
});
