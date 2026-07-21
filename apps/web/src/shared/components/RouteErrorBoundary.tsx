import React from 'react';
import { useRouteError, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { AlertTriangle, Home, RefreshCcw } from 'lucide-react';
import { GlassCard } from './GlassCard';

export const RouteErrorBoundary = () => {
  const error = useRouteError() as Error;
  const navigate = useNavigate();

  return (
    <div className="min-h-screen w-full bg-[var(--bg-1)] flex items-center justify-center p-6 bg-[radial-gradient(ellipse_at_top,_var(--bg-2)_0%,_var(--bg-1)_100%)]">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-lg"
      >
        <GlassCard className="p-8 border-[var(--state-danger)]/30 border-2 text-center">
          <div className="w-20 h-20 mx-auto rounded-full bg-[var(--state-danger)]/10 flex items-center justify-center mb-6">
            <AlertTriangle className="w-10 h-10 text-[var(--state-danger)]" />
          </div>
          
          <h1 className="text-3xl font-[var(--font-display)] font-bold text-[var(--text-primary)] mb-4">
            System Error
          </h1>
          
          <div className="bg-[var(--bg-2)] rounded-lg p-4 mb-8 border border-[var(--bg-3)]">
            <p className="text-[var(--text-secondary)] font-[var(--font-mono)] text-sm whitespace-pre-wrap break-words text-left">
              {error?.message || 'An unexpected application error occurred.'}
            </p>
          </div>
          
          <div className="flex gap-4 justify-center">
            <button 
              onClick={() => window.location.reload()}
              className="px-6 py-2.5 rounded-lg font-[var(--font-body)] text-sm font-medium flex items-center gap-2 bg-[var(--bg-2)] hover:bg-[var(--bg-3)] text-[var(--text-primary)] transition-colors border border-[var(--bg-3)]"
            >
              <RefreshCcw className="w-4 h-4" /> Reload Page
            </button>
            <button 
              onClick={() => navigate('/')}
              className="px-6 py-2.5 rounded-lg font-[var(--font-body)] text-sm font-medium flex items-center gap-2 bg-[var(--accent-primary)] text-[var(--bg-1)] hover:opacity-90 transition-opacity"
            >
              <Home className="w-4 h-4" /> Go to Dashboard
            </button>
          </div>
        </GlassCard>
      </motion.div>
    </div>
  );
};
