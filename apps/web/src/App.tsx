import React from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { LoginPage } from './components/auth/LoginPage';
import { ResearcherPortal } from './components/researcher/ResearcherPortal';
import { NeuroAegisDashboard } from './pages/NeuroAegisDashboard';
import { Activity, Loader2 } from 'lucide-react';

const AppContent: React.FC = () => {
  const { status, user } = useAuth();

  if (status === 'loading') {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center text-slate-300">
        <div className="h-12 w-12 rounded-2xl bg-indigo-600 flex items-center justify-center shadow-xl shadow-indigo-500/25 mb-4 animate-pulse">
          <Activity className="h-6 w-6 text-white" />
        </div>
        <div className="flex items-center space-x-2 text-sm font-medium text-slate-400">
          <Loader2 className="h-4 w-4 animate-spin text-indigo-400" />
          <span>Verifying institutional session...</span>
        </div>
      </div>
    );
  }

  if (status === 'unauthenticated' || !user) {
    return <LoginPage />;
  }

  if (user.role === 'researcher') {
    return <ResearcherPortal />;
  }

  // Admin and Clinician access the full clinical telemetry dashboard
  return <NeuroAegisDashboard />;
};

export const App: React.FC = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

export default App;
