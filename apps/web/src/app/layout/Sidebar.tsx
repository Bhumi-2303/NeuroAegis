import { Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Brain, 
  Activity, 
  BarChart2, 
  Cpu, 
  Users, 
  Settings 
} from 'lucide-react';
import { GlassPanel } from '../../design-system/primitives';

const NAV_ITEMS = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/analysis', icon: Brain, label: 'Brain Analysis' },
  { path: '/eeg', icon: Activity, label: 'Neural Activity' },
  { path: '/explainability', icon: BarChart2, label: 'Explainability' },
  { path: '/prediction', icon: Cpu, label: 'AI Models' },
  { path: '/patients', icon: Users, label: 'Patients' },
  { path: '/settings', icon: Settings, label: 'Settings' },
];

export const Sidebar = () => {
  const location = useLocation();

  return (
    <GlassPanel className="hidden md:flex fixed left-0 top-[72px] bottom-0 w-[64px] flex-col items-center py-6 gap-6 rounded-none border-y-0 border-l-0 z-40 transition-all duration-300">
      {NAV_ITEMS.map((item) => {
        const isActive = location.pathname === item.path || 
                         (item.path !== '/' && location.pathname.startsWith(item.path));
        const Icon = item.icon;

        return (
          <Link
            key={item.path}
            to={item.path}
            className={`relative group p-2 rounded-xl transition-all duration-300 ${
              isActive 
                ? 'text-[var(--accent-primary)] bg-[rgba(0,229,255,0.08)]' 
                : 'text-[var(--text-secondary)] hover:text-[var(--accent-primary)] hover:bg-[rgba(255,255,255,0.04)]'
            }`}
          >
            <Icon size={20} strokeWidth={1.5} />
            
            {/* Tooltip */}
            <div className="absolute left-full top-1/2 -translate-y-1/2 ml-4 px-3 py-1.5 rounded-lg opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity bg-[rgba(11,22,37,0.8)] backdrop-blur-md border border-[rgba(255,255,255,0.08)] text-xs whitespace-nowrap z-50">
              {item.label}
            </div>
            
            {isActive && (
              <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-[var(--accent-primary)] rounded-r-full shadow-[0_0_10px_var(--accent-primary)]" />
            )}
          </Link>
        );
      })}
    </GlassPanel>
  );
};
