import type { Alert } from '@neuroaegis/model-contracts';

export function generateMockAlerts(): Alert[] {
  const now = Date.now();
  
  return [
    {
      id: 'alert-1',
      severity: 'critical',
      message: 'Seizure event detected — session 4821 (F7-T3 high variance)',
      createdAt: new Date(now - 120000).toISOString()
    },
    {
      id: 'alert-2',
      severity: 'warning',
      message: 'Elevated theta activity in frontal channels',
      createdAt: new Date(now - 360000).toISOString()
    },
    {
      id: 'alert-3',
      severity: 'info',
      message: 'Session 4821 model inference latency spike (>50ms)',
      createdAt: new Date(now - 720000).toISOString()
    }
  ];
}
