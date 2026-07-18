
import { motion } from 'framer-motion';
import { Users, FileX, AlertCircle } from 'lucide-react';
import { useAlerts } from '../hooks/useAlerts';
import { GlassCard, SkeletonShimmer, EmptyState, ErrorState } from '../../../shared/components';
import { pageTransition, fadeIn } from '../../../shared/lib/motion-presets';

export const PatientsPage = () => {
  const { data, isLoading, isError, error, refetch } = useAlerts();

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
      <header className="flex items-center gap-3 mb-8">
        <Users className="w-8 h-8 text-[var(--accent-primary)]" />
        <h1 className="text-3xl font-display font-bold text-[var(--text-primary)]">Patient Management</h1>
      </header>

      <GlassCard title="System Alerts" className="mb-6">
        <div className="p-4">
          {renderAlertsContent()}
        </div>
      </GlassCard>

      <GlassCard>
        <div className="p-8">
          <EmptyState
            icon={FileX}
            title="Records Unavailable"
            description="Patient records will be available when backend integration is complete."
          />
        </div>
      </GlassCard>
    </motion.div>
  );
};
