import { createBrowserRouter, Navigate } from 'react-router-dom';
import { DashboardShell } from './layout';

// Feature Pages
import { DashboardPage } from '../features/dashboard';
import { BrainAnalysisPage } from '../features/brain-analysis';
import { EEGMonitorPage } from '../features/eeg-monitor';
import { FrequencyAnalysisPage } from '../features/frequency-analysis';

import { ExplainabilityPage } from '../features/explainability';
import { ReportsPage } from '../features/reports';
import { PatientsPage } from '../features/patients';
import { SettingsPage } from '../features/settings';

// Shared Components
import { RouteErrorBoundary } from '../shared/components/RouteErrorBoundary';

// Doctor V2
import { DoctorDashboard } from '../doctor/pages/DoctorDashboard';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <DashboardShell />,
    errorElement: <RouteErrorBoundary />,
    children: [
      {
        index: true,
        element: <DashboardPage />,
      },
      {
        path: 'analysis',
        element: <BrainAnalysisPage />,
      },
      {
        path: 'eeg',
        element: <EEGMonitorPage />,
      },
      {
        path: 'frequency',
        element: <FrequencyAnalysisPage />,
      },
      {
        path: 'prediction',
        element: <Navigate to="/" replace />,
      },
      {
        path: 'explainability',
        element: <ExplainabilityPage />,
      },
      {
        path: 'reports',
        element: <ReportsPage />,
      },
      {
        path: 'patients',
        element: <PatientsPage />,
      },
      {
        path: 'settings',
        element: <SettingsPage />,
      },
    ],
  },
  {
    path: '/doctor',
    element: <DoctorDashboard />
  }
]);
