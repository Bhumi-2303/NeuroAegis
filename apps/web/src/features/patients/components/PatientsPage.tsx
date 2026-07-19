import { useState } from 'react';
import { motion } from 'framer-motion';
import { Users, FileX, AlertCircle, History } from 'lucide-react';
import { useAlerts } from '../hooks/useAlerts';
import { 
  GlassCard, 
  SkeletonShimmer, 
  EmptyState, 
  ErrorState,
  HudCornerFrame,
  SessionFilmstrip,
  ClinicalMetaTable
} from '../../../shared/components';
import { pageTransition, fadeIn, staggerChildren, slideUp } from '../../../shared/lib/motion-presets';

const MOCK_SESSIONS = [
  { id: 'sess-001', label: 'Session 001', timestamp: '10:30 AM' },
  { id: 'sess-002', label: 'Session 002', timestamp: '11:15 AM' },
  { id: 'sess-003', label: 'Session 003', timestamp: '01:45 PM' },
  { id: 'sess-004', label: 'Session 004', timestamp: '03:20 PM' },
  { id: 'sess-005', label: 'Session 005', timestamp: '04:00 PM' },
];

export const PatientsPage = () => {
  const { data, isLoading, isError, error, refetch } = useAlerts();
  const [activeSession, setActiveSession] = useState('sess-003');

  const renderAlertsContent = () => {
    if (isLoading) {
      return (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <SkeletonShimmer key={i} className="h-16 w-full rounded-lg" />
          ))}
        </div>
      );
    }

    if (isError) {
      return (
        <ErrorState
          title="Failed to load alerts"
          message={error instanceof Error ? error.message : 'An unknown error occurred'}
          onRetry={refetch}
        />
      );
    }

    if (!data || data.length === 0) {
      return (
        <EmptyState
          icon={AlertCircle}
          title="No Alerts"
          description="There are currently no active system alerts."
        />
      );
    }

    return (
      <motion.div variants={fadeIn} initial="initial" animate="animate" className="space-y-3">
        {data.map((alert) => {
          let dotColor = 'var(--accent-secondary)';
          if (alert.severity === 'warning') dotColor = 'var(--state-warning)';
          if (alert.severity === 'critical') dotColor = 'var(--state-danger)';

          return (
            <div key={alert.id} className="flex items-center gap-4 p-3 rounded-lg bg-[var(--bg-1)] border border-[var(--bg-3)]">
              <div 
                className="w-3 h-3 rounded-full shrink-0" 
                style={{ backgroundColor: dotColor, boxShadow: `0 0 10px ${dotColor}` }} 
              />
              <div className="flex-1 font-body text-sm text-[var(--text-primary)]">
                {alert.message}
              </div>
              <div className="font-mono text-xs text-[var(--text-secondary)] whitespace-nowrap">
                {alert.createdAt}
              </div>
            </div>
          );
        })}
      </motion.div>
    );
  };

  return (
    <motion.div variants={pageTransition} initial="initial" animate="animate" exit="exit" className="p-6 max-w-6xl mx-auto space-y-6">
      <header className="mb-8">
        <HudCornerFrame label="Patient Management" icon={<Users size={14} strokeWidth={1.5} />}>
          <h1 className="text-3xl font-display font-bold text-[var(--text-primary)]">Patient Management</h1>
        </HudCornerFrame>
      </header>

      <motion.div variants={staggerChildren} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <motion.div variants={slideUp} className="lg:col-span-2">
          <GlassCard className="h-full">
            <HudCornerFrame label="System Alerts">
              {renderAlertsContent()}
            </HudCornerFrame>
          </GlassCard>
        </motion.div>

        <motion.div variants={slideUp} className="lg:col-span-1 flex flex-col gap-6">
          <GlassCard>
            <HudCornerFrame label="Recent Sessions" icon={<History size={12} strokeWidth={1.5} />}>
              <SessionFilmstrip
                sessions={MOCK_SESSIONS}
                activeId={activeSession}
                onSelect={setActiveSession}
              />
            </HudCornerFrame>
          </GlassCard>

          <GlassCard>
            <HudCornerFrame label="Session Metadata">
              <ClinicalMetaTable
                rows={[
                  { attribute: 'Session ID', value: activeSession },
                  { attribute: 'Patient ID', value: 'PT-8842' },
                  { attribute: 'Duration', value: '45m 12s' },
                  { attribute: 'Avg. Confidence', value: '92%' },
                  { attribute: 'Notes', value: 'Routine' }
                ]}
              />
            </HudCornerFrame>
          </GlassCard>
        </motion.div>

        <motion.div variants={slideUp} className="lg:col-span-3 mt-2">
          <GlassCard>
            <HudCornerFrame label="Patient Records">
              <EmptyState
                icon={FileX}
                title="Records Unavailable"
                description="Patient records will be available when backend integration is complete."
              />
            </HudCornerFrame>
          </GlassCard>
        </motion.div>
      </motion.div>
    </motion.div>
  );
};
