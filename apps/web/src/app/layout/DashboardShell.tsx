import { Outlet } from 'react-router-dom';
import { motion } from 'framer-motion';
import { TopNav } from './TopNav';
import { Sidebar } from './Sidebar';
import { GlassCard } from '../../shared/components';
import { Activity, Radio, Brain, Lightbulb, Zap, Timer, AlertTriangle, Signal } from 'lucide-react';
import { useDashboardAlerts } from '../../features/dashboard/hooks/useDashboardData';
import { staggerChildren } from '../../shared/lib/motion-presets';

// ── Bottom Metric Cards ──
interface MetricCardProps {
  label: string;
  value: string;
  icon: React.ReactNode;
  accentColor: string;
}

function MetricCard({ label, value, icon, accentColor }: MetricCardProps): React.JSX.Element {
  return (
    <GlassCard className="flex flex-row items-center gap-3 p-3 min-w-0">
      <div
        className="flex items-center justify-center w-10 h-10 rounded-xl shrink-0"
        style={{ backgroundColor: `color-mix(in srgb, ${accentColor} 12%, transparent)` }}
      >
        <span style={{ color: accentColor }}>{icon}</span>
      </div>
      <div className="flex flex-col min-w-0">
        <span className="text-xs text-[var(--text-secondary)] truncate">{label}</span>
        <span className="text-lg font-semibold font-mono text-[var(--text-primary)]">{value}</span>
      </div>
    </GlassCard>
  );
}

// ── Quick-Link Nav Card ──
interface QuickLinkProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  href: string;
  accentColor: string;
}

function QuickLinkCard({ title, description, icon, href, accentColor }: QuickLinkProps): React.JSX.Element {
  return (
    <a href={href} className="block no-underline group">
      <GlassCard className="h-full transition-all group-hover:border-[rgba(0,229,255,0.2)]">
        <div className="flex items-start gap-3">
          <div
            className="flex items-center justify-center w-10 h-10 rounded-xl shrink-0"
            style={{ backgroundColor: `color-mix(in srgb, ${accentColor} 12%, transparent)` }}
          >
            <span style={{ color: accentColor }}>{icon}</span>
          </div>
          <div className="flex flex-col gap-1 min-w-0">
            <h4 className="text-sm font-semibold text-[var(--text-primary)] m-0">{title}</h4>
            <p className="text-xs text-[var(--text-secondary)] m-0 line-clamp-2">{description}</p>
          </div>
        </div>
      </GlassCard>
    </a>
  );
}

export const DashboardShell = () => {
  const { criticalCount: alertCount } = useDashboardAlerts();

  return (
    <div className="relative min-h-screen w-full flex flex-col">
      <TopNav />
      <div className="flex flex-1 pt-[72px]">
        <Sidebar />
        <main className="flex-1 ml-0 md:ml-[64px] flex flex-col min-w-0 transition-all duration-300">
          <div className="flex-1 p-4 md:p-6 overflow-y-auto">
            <Outlet />
          </div>

          {/* Persistent Footer: Metrics and Links */}
          <div className="border-t border-[rgba(255,255,255,0.05)] bg-[rgba(11,22,37,0.5)] backdrop-blur-md p-4 md:p-6 flex flex-col gap-6 z-10">
            {/* Bottom Metric Cards */}
            <motion.div
              className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3"
              role="region"
              aria-label="System Metrics"
              {...staggerChildren}
            >
              <MetricCard
                label="Signal Strength"
                value="87%"
                icon={<Signal size={18} strokeWidth={1.5} />}
                accentColor="var(--state-success)"
              />
              <MetricCard
                label="Active Channels"
                value="4"
                icon={<Radio size={18} strokeWidth={1.5} />}
                accentColor="var(--accent-primary)"
              />
              <MetricCard
                label="Confidence"
                value="94%"
                icon={<Zap size={18} strokeWidth={1.5} />}
                accentColor="var(--accent-highlight)"
              />
              <MetricCard
                label="Session Duration"
                value="12:34"
                icon={<Timer size={18} strokeWidth={1.5} />}
                accentColor="var(--accent-secondary)"
              />
              <MetricCard
                label="Alert Count"
                value={String(alertCount)}
                icon={<AlertTriangle size={18} strokeWidth={1.5} />}
                accentColor={alertCount > 0 ? 'var(--state-danger)' : 'var(--state-success)'}
              />
            </motion.div>

            {/* Quick Links */}
            <div role="navigation" aria-label="System Modules">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <QuickLinkCard
                  title="EEG Monitor"
                  description="Real-time neural activity waveform visualization"
                  icon={<Activity size={18} strokeWidth={1.5} />}
                  href="/eeg"
                  accentColor="var(--accent-primary)"
                />
                <QuickLinkCard
                  title="Frequency Analysis"
                  description="Gamma, Beta, Alpha, Theta, Delta band power"
                  icon={<Radio size={18} strokeWidth={1.5} />}
                  href="/frequency"
                  accentColor="var(--state-success)"
                />
                <QuickLinkCard
                  title="Seizure Prediction"
                  description="Per-model classification with confidence scores"
                  icon={<Brain size={18} strokeWidth={1.5} />}
                  href="/prediction"
                  accentColor="var(--accent-highlight)"
                />
                <QuickLinkCard
                  title="Explainability"
                  description="SHAP-based feature attribution analysis"
                  icon={<Lightbulb size={18} strokeWidth={1.5} />}
                  href="/explainability"
                  accentColor="var(--state-warning)"
                />
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};
