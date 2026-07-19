import { motion } from 'framer-motion';
import {
  Activity,
  Radio,
  Brain,
  Lightbulb,
  Zap,
  Timer,
  AlertTriangle,
  Signal,
  Move,
  ZoomIn,
  Ruler,
  Sparkles,
} from 'lucide-react';
import { GlassCard, SkeletonShimmer, WaveformSparkline, Scene, HudCornerFrame } from '../../../shared/components';
import { slideUp, staggerChildren } from '../../../shared/lib/motion-presets';
import { useDashboardEEG, useDashboardAlerts } from '../hooks/useDashboardData';

// ── Bottom Metric Cards ──
interface MetricCardProps {
  label: string;
  value: string;
  icon: React.ReactNode;
  accentColor: string;
}

function MetricCard({ label, value, icon, accentColor }: MetricCardProps): React.JSX.Element {
  return (
    <div className="flex flex-row items-center gap-3 p-3 min-w-0 relative z-10">
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
    </div>
  );
}

// ── Toolbar ──
function EEGToolbar(): React.JSX.Element {
  return (
    <div className="flex items-center gap-1 px-3 py-1.5 rounded-full bg-[rgba(11,22,37,0.7)] border border-[rgba(255,255,255,0.06)] backdrop-blur-sm">
      <button
        className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:text-[var(--accent-primary)] hover:bg-[rgba(0,229,255,0.06)] transition-colors"
        aria-label="Pan"
        title="Pan"
      >
        <Move size={14} strokeWidth={1.5} />
      </button>
      <button
        className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:text-[var(--accent-primary)] hover:bg-[rgba(0,229,255,0.06)] transition-colors"
        aria-label="Zoom"
        title="Zoom"
      >
        <ZoomIn size={14} strokeWidth={1.5} />
      </button>
      <button
        className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:text-[var(--accent-primary)] hover:bg-[rgba(0,229,255,0.06)] transition-colors"
        aria-label="Measure"
        title="Measure"
      >
        <Ruler size={14} strokeWidth={1.5} />
      </button>

      <div className="w-px h-4 bg-[rgba(255,255,255,0.08)] mx-1" />

      <span className="text-[10px] font-mono text-[var(--text-secondary)] px-1">100%</span>

      <div className="w-px h-4 bg-[rgba(255,255,255,0.08)] mx-1" />

      <button
        className="flex items-center gap-1 px-2 py-1 rounded-full text-[10px] font-mono bg-[rgba(139,92,246,0.1)] text-[var(--accent-highlight)] border border-[rgba(139,92,246,0.2)] hover:bg-[rgba(139,92,246,0.2)] transition-colors"
        aria-label="AI Explain"
      >
        <Sparkles size={10} strokeWidth={1.5} />
        AI Explain
      </button>
    </div>
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

export function DashboardPage(): React.JSX.Element {
  const { sparklineValues: sparklineData, isLoading: eegLoading } = useDashboardEEG(['Fp1', 'Fp2', 'C3', 'C4']);
  const { criticalCount: alertCount } = useDashboardAlerts();

  return (
    <motion.div
      className="flex flex-col gap-6 p-6"
      {...slideUp}
    >
      {/* Page Title with HudCornerFrame */}
      <HudCornerFrame label="Command Center" icon={<Brain size={14} strokeWidth={1.5} />}>
        <h1 id="dashboard-title" className="text-2xl font-display font-bold text-[var(--text-primary)] m-0">
          NeuroAegis Command Center
        </h1>
        <p className="text-sm text-[var(--text-secondary)] m-0 mt-0.5">
          AI-Powered Epileptic Seizure Detection System
        </p>
      </HudCornerFrame>

      {/* Top: Hero Placeholder + Neural Signal */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Hero Brain 3D Scene */}
        <GlassCard
          className="lg:col-span-2 h-[320px] p-0 overflow-hidden relative flex flex-col"
          role="region"
          aria-label="3D Brain Analysis View"
        >
          <div className="absolute top-4 left-4 flex items-center gap-2 z-10">
             <Brain size={16} strokeWidth={1.5} className="text-[var(--text-secondary)]" />
             <span className="text-sm font-semibold text-[var(--text-secondary)]">Brain Analysis</span>
          </div>
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#8B5CF6]/10 via-transparent to-transparent pointer-events-none" />
          <Scene interactive={false} />
        </GlassCard>

        {/* Neural Signal Timeline */}
        <GlassCard
          className="min-h-[320px]"
          role="region"
          aria-label="Live EEG Neural Signal Monitor"
        >
          <HudCornerFrame label="Neural Signal" icon={<Activity size={12} strokeWidth={1.5} />}>
            {/* Toolbar */}
            <div className="mb-3">
              <EEGToolbar />
            </div>

            <div className="flex flex-col gap-4 flex-1 justify-between" aria-live="polite">
              {eegLoading ? (
                <>
                  <SkeletonShimmer className="h-16 rounded-lg" />
                  <SkeletonShimmer className="h-16 rounded-lg" />
                  <SkeletonShimmer className="h-16 rounded-lg" />
                </>
              ) : (
                <>
                  <div className="flex flex-col gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Live Waveform — Fp1</span>
                    <WaveformSparkline data={sparklineData} className="h-16" />
                  </div>
                  <div className="flex flex-col gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Signal Quality</span>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 rounded-full bg-[rgba(255,255,255,0.06)] overflow-hidden" role="progressbar" aria-valuenow={87} aria-valuemin={0} aria-valuemax={100} aria-label="Signal Quality">
                        <div
                          className="h-full rounded-full transition-all duration-700"
                          style={{
                            width: '87%',
                            background: 'linear-gradient(90deg, var(--accent-primary), var(--state-success))'
                          }}
                        />
                      </div>
                      <span className="text-xs font-mono text-[var(--state-success)]" aria-hidden="true">87%</span>
                    </div>
                  </div>
                  <div className="flex flex-col gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Active Channels</span>
                    <div className="flex gap-1.5">
                      {['Fp1', 'Fp2', 'C3', 'C4'].map((ch) => (
                        <span
                          key={ch}
                          className="px-2 py-0.5 rounded-md text-[10px] font-mono
                                     bg-[rgba(0,229,255,0.08)] text-[var(--accent-primary)]
                                     border border-[rgba(0,229,255,0.15)]"
                        >
                          {ch}
                        </span>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>
          </HudCornerFrame>
        </GlassCard>
      </div>

      {/* Bottom Metric Cards with continuous gradient strip */}
      <motion.div
        className="relative rounded-2xl overflow-hidden"
        role="region"
        aria-label="System Metrics"
        {...staggerChildren}
      >
        {/* Continuous multi-color gradient strip running underneath all cards */}
        <div
          className="absolute bottom-0 left-0 right-0 h-1 rounded-full"
          style={{
            background: 'linear-gradient(90deg, #00FFA3, #00E5FF, #8B5CF6, #FFB020, #FF4D6D)',
            boxShadow: '0 0 12px rgba(0, 229, 255, 0.2)',
          }}
        />
        <div
          className="absolute bottom-0 left-0 right-0 h-8 pointer-events-none"
          style={{
            background: 'linear-gradient(0deg, rgba(0,229,255,0.03), transparent)',
          }}
        />

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-px bg-[rgba(255,255,255,0.04)]">
          <div className="bg-[rgba(11,22,37,0.6)] backdrop-blur-sm">
            <MetricCard
              label="Signal Strength"
              value="87%"
              icon={<Signal size={18} strokeWidth={1.5} />}
              accentColor="var(--state-success)"
            />
          </div>
          <div className="bg-[rgba(11,22,37,0.6)] backdrop-blur-sm">
            <MetricCard
              label="Active Channels"
              value="4"
              icon={<Radio size={18} strokeWidth={1.5} />}
              accentColor="var(--accent-primary)"
            />
          </div>
          <div className="bg-[rgba(11,22,37,0.6)] backdrop-blur-sm">
            <MetricCard
              label="Confidence"
              value="94%"
              icon={<Zap size={18} strokeWidth={1.5} />}
              accentColor="var(--accent-highlight)"
            />
          </div>
          <div className="bg-[rgba(11,22,37,0.6)] backdrop-blur-sm">
            <MetricCard
              label="Session Duration"
              value="12:34"
              icon={<Timer size={18} strokeWidth={1.5} />}
              accentColor="var(--accent-secondary)"
            />
          </div>
          <div className="bg-[rgba(11,22,37,0.6)] backdrop-blur-sm">
            <MetricCard
              label="Alert Count"
              value={String(alertCount)}
              icon={<AlertTriangle size={18} strokeWidth={1.5} />}
              accentColor={alertCount > 0 ? 'var(--state-danger)' : 'var(--state-success)'}
            />
          </div>
        </div>
      </motion.div>

      {/* Quick Links */}
      <div role="navigation" aria-label="System Modules">
        <HudCornerFrame label="System Modules">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mt-2">
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
        </HudCornerFrame>
      </div>
    </motion.div>
  );
}
