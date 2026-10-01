import React from 'react';
import { useAuth } from '../../context/AuthContext';
import {
  Activity,
  LogOut,
  ShieldAlert,
  Brain,
  Cpu,
  BarChart3,
  Layers,
  FileCode,
  CheckCircle2,
} from 'lucide-react';

export const ResearcherPortal: React.FC = () => {
  const { user, logout, logoutAll } = useAuth();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-white">
      {/* Top Navigation */}
      <header className="bg-slate-900/95 backdrop-blur border-b border-slate-800 px-4 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30 shadow-md">
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-lg bg-cyan-600 flex items-center justify-center shadow-lg shadow-cyan-500/25">
            <Brain className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-base lg:text-lg font-bold text-white tracking-wide">
                NEURO<span className="text-cyan-400">AEGIS</span>
              </h1>
              <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-700/50">
                RESEARCH PORTAL
              </span>
            </div>
            <p className="text-[11px] text-slate-400">Model Architectures & De-identified Benchmark Telemetry</p>
          </div>
        </div>

        {/* User Badge & Actions */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-slate-950/80 border border-slate-800 rounded-lg px-3 py-1.5 text-xs">
            <div className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="font-semibold text-slate-200">{user?.username}</span>
            <span className="text-[10px] uppercase px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
              {user?.role}
            </span>
            <span className="text-[10px] font-mono text-slate-500">
              [{user?.tenant_id}]
            </span>
          </div>

          <button
            type="button"
            onClick={() => logout()}
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-white text-xs font-medium flex items-center space-x-1.5 transition"
          >
            <LogOut className="h-3.5 w-3.5" />
            <span>Sign Out</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-8 space-y-6">
        {/* PHI Boundary Notice Banner */}
        <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/60 text-amber-200 flex items-start space-x-3.5 shadow-lg">
          <ShieldAlert className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1 text-xs">
            <h2 className="font-bold text-sm text-amber-300">
              Clinical PHI & Live Telemetry Access Boundary
            </h2>
            <p className="text-amber-200/90 leading-relaxed">
              Under institutional privacy boundaries and multi-tenant clinical data isolation policy,
              the <strong>Researcher</strong> role is isolated from identifiable patient records, clinical patient modification,
              and live bedside telemetry dispatch. All model metrics, dataset benchmarks, and explainability representations below
              are derived strictly from de-identified research cohorts.
            </p>
          </div>
        </div>

        {/* Overview Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs uppercase font-semibold">Active Architecture</span>
              <Cpu className="h-4 w-4 text-cyan-400" />
            </div>
            <div className="text-xl font-bold text-white">GNN + STFT</div>
            <div className="text-[11px] text-slate-500 mt-1">Spatial Graph Neural Network</div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs uppercase font-semibold">Bonn Benchmark</span>
              <BarChart3 className="h-4 w-4 text-emerald-400" />
            </div>
            <div className="text-xl font-bold text-emerald-400">98.4% Acc</div>
            <div className="text-[11px] text-slate-500 mt-1">Sensitivity 97.8% | Spec 99.1%</div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs uppercase font-semibold">CHB-MIT Benchmark</span>
              <BarChart3 className="h-4 w-4 text-cyan-400" />
            </div>
            <div className="text-xl font-bold text-cyan-400">92.1% Sens</div>
            <div className="text-[11px] text-slate-500 mt-1">FPR 0.12/h pediatric scalp</div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs uppercase font-semibold">Inference Latency</span>
              <Activity className="h-4 w-4 text-purple-400" />
            </div>
            <div className="text-xl font-bold text-purple-400">&lt; 420 ms</div>
            <div className="text-[11px] text-slate-500 mt-1">End-to-end SHAP + GNN</div>
          </div>
        </div>

        {/* Architecture & Pipeline Specifications */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex items-center space-x-2 pb-3 border-b border-slate-800">
              <Layers className="h-4 w-4 text-cyan-400" />
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                Graph Neural Network Pipeline
              </h2>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              NeuroAegis constructs dynamic graph representations where nodes represent 10-20 international
              electrode channels and edge weights encode instantaneous phase-locking value (PLV) and coherence matrices.
            </p>
            <ul className="space-y-2 text-xs text-slate-300">
              <li className="flex items-center space-x-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                <span>Short-Time Fourier Transform (STFT) spectral decomposition (0.5–60 Hz)</span>
              </li>
              <li className="flex items-center space-x-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                <span>Multi-scale spatial Chebyshev graph convolutions</span>
              </li>
              <li className="flex items-center space-x-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                <span>Temporal bidirectional LSTM attention pooling</span>
              </li>
              <li className="flex items-center space-x-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                <span>Calibrated Platt-scaling posterior confidence bands</span>
              </li>
            </ul>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex items-center space-x-2 pb-3 border-b border-slate-800">
              <FileCode className="h-4 w-4 text-cyan-400" />
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                Explainable AI (SHAP) Model Attribution
              </h2>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Prediction explanations are computed via Kernel and Tree SHAP engines operating on spectral entropy,
              delta/theta/alpha/beta power ratios, and regional inter-electrode synchronization.
            </p>
            <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800/80 space-y-2 text-xs font-mono">
              <div className="flex justify-between text-slate-400">
                <span>theta_band_power (F3-C3)</span>
                <span className="text-rose-400">+0.342 SHAP</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>spectral_entropy (T3-T5)</span>
                <span className="text-rose-400">+0.281 SHAP</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>alpha_attenuation (O1-O2)</span>
                <span className="text-rose-400">+0.194 SHAP</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>background_rhythm_stability</span>
                <span className="text-emerald-400">-0.118 SHAP</span>
              </div>
            </div>
          </div>
        </div>

        {/* Session Security Footprint */}
        <div className="bg-slate-900/50 border border-slate-800/80 rounded-xl p-4 flex flex-wrap items-center justify-between text-xs text-slate-400 gap-3">
          <div className="flex items-center space-x-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <span>Authenticated via secure HttpOnly cookie session</span>
          </div>
          <button
            type="button"
            onClick={() => logoutAll()}
            className="text-rose-400 hover:text-rose-300 hover:underline transition"
          >
            Revoke all active sessions on this account
          </button>
        </div>
      </main>
    </div>
  );
};
