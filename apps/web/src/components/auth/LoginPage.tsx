import React, { useState, FormEvent } from 'react';
import { useAuth } from '../../context/AuthContext';
import { Activity, Lock, User, AlertCircle, ShieldCheck, ArrowRight, Loader2 } from 'lucide-react';

export const LoginPage: React.FC = () => {
  const { login, error, clearError } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password) {
      setLocalError('Please enter both username and password.');
      return;
    }

    setLocalError(null);
    clearError();
    setIsSubmitting(true);

    try {
      await login(username.trim(), password);
    } catch (err) {
      setPassword(''); // Clear password on error
      const msg = err instanceof Error ? err.message : 'Authentication failed';
      setLocalError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleQuickFill = (demoUser: string) => {
    setUsername(demoUser);
    setPassword('password123');
    setLocalError(null);
    clearError();
  };

  const displayError = localError || error;

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col justify-center items-center px-4 py-12 selection:bg-indigo-500 selection:text-white">
      {/* Decorative background glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 -right-40 w-96 h-96 bg-cyan-600/10 rounded-full blur-3xl" />
      </div>

      <div className="w-full max-w-md relative z-10">
        {/* Brand Header */}
        <div className="text-center mb-8">
          <div className="inline-flex h-14 w-14 rounded-2xl bg-indigo-600 items-center justify-center shadow-xl shadow-indigo-500/25 mb-4 border border-indigo-400/30">
            <Activity className="h-8 w-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            NEURO<span className="text-indigo-400">AEGIS</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1 uppercase tracking-widest font-semibold">
            Clinical EEG Telemetry & Decision Support
          </p>
        </div>

        {/* Card */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-2xl backdrop-blur-md">
          <div className="flex items-center space-x-2 pb-6 border-b border-slate-800/80 mb-6">
            <ShieldCheck className="h-5 w-5 text-indigo-400" />
            <span className="text-sm font-semibold text-slate-200">
              Institutional Session Sign-In
            </span>
          </div>

          {displayError && (
            <div
              role="alert"
              className="mb-6 p-3.5 rounded-xl bg-rose-950/50 border border-rose-800/60 text-rose-300 text-xs flex items-start space-x-2.5"
            >
              <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
              <span>{displayError}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4" noValidate>
            <div>
              <label
                htmlFor="username"
                className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5"
              >
                Username or Clinical ID
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
                  <User className="h-4 w-4" />
                </div>
                <input
                  id="username"
                  name="username"
                  type="text"
                  autoComplete="username"
                  required
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="e.g. clinician_a"
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-2.5 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition"
                  disabled={isSubmitting}
                />
              </div>
            </div>

            <div>
              <label
                htmlFor="password"
                className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5"
              >
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
                  <Lock className="h-4 w-4" />
                </div>
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••••"
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-2.5 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition"
                  disabled={isSubmitting}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full mt-2 py-3 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-600/50 text-white font-semibold text-sm shadow-lg shadow-indigo-600/25 transition flex items-center justify-center space-x-2 cursor-pointer disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Verifying Session...</span>
                </>
              ) : (
                <>
                  <span>Sign In</span>
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </button>
          </form>

          {/* Institutional Credentials Quick-Select */}
          <div className="mt-8 pt-6 border-t border-slate-800 text-xs">
            <span className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2.5 text-center">
              Testing & Institutional Demo Roles
            </span>
            <div className="grid grid-cols-3 gap-2">
              <button
                type="button"
                onClick={() => handleQuickFill('clinician_a')}
                className="p-2 rounded-lg bg-slate-950/80 hover:bg-slate-800/80 border border-slate-800 text-left transition text-slate-300 hover:text-white"
              >
                <div className="font-semibold text-indigo-400">Clinician</div>
                <div className="text-[10px] text-slate-500 font-mono">clinician_a</div>
              </button>
              <button
                type="button"
                onClick={() => handleQuickFill('admin_a')}
                className="p-2 rounded-lg bg-slate-950/80 hover:bg-slate-800/80 border border-slate-800 text-left transition text-slate-300 hover:text-white"
              >
                <div className="font-semibold text-purple-400">Admin</div>
                <div className="text-[10px] text-slate-500 font-mono">admin_a</div>
              </button>
              <button
                type="button"
                onClick={() => handleQuickFill('researcher_a')}
                className="p-2 rounded-lg bg-slate-950/80 hover:bg-slate-800/80 border border-slate-800 text-left transition text-slate-300 hover:text-white"
              >
                <div className="font-semibold text-cyan-400">Researcher</div>
                <div className="text-[10px] text-slate-500 font-mono">researcher_a</div>
              </button>
            </div>
          </div>
        </div>

        {/* Security Notice */}
        <p className="text-center text-[11px] text-slate-500 mt-6">
          Protected health information system. All access is logged and audited under institutional policy.
        </p>
      </div>
    </div>
  );
};
