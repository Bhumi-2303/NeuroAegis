import { Outlet } from 'react-router-dom';
import { motion } from 'framer-motion';
import { TopNav } from './TopNav';
import { Sidebar } from './Sidebar';
import { GlassCard } from '../../shared/components';
import { Activity, Brain, Signal } from 'lucide-react';
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

export const DashboardShell = () => {
  return (
    <div className="relative min-h-screen w-full flex flex-col bg-[var(--bg-1)]">
      {/* Banner */}
      <div className="fixed top-0 left-0 right-0 z-[60] h-8 flex items-center justify-center bg-[var(--state-warning)] text-[var(--bg-2)] text-xs font-semibold uppercase tracking-wider">
        ⚠️ Research prototype — not for clinical use
      </div>
      
      <TopNav />
      
      <div className="flex flex-1 pt-[104px]"> {/* 72px TopNav + 32px banner */}
        <Sidebar />
        <main className="flex-1 ml-0 md:ml-[64px] flex flex-col min-w-0 transition-all duration-300">
          <div className="flex-1 p-4 md:p-6 overflow-y-auto">
            <Outlet />
          </div>

        </main>
      </div>
    </div>
  );
};
