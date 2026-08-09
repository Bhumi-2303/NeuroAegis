import { Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Brain, 
  Activity
} from 'lucide-react';

const NAV_ITEMS = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/analysis', icon: Brain, label: 'Brain Analysis' },
  { path: '/eeg', icon: Activity, label: 'Neural Activity' },
];

export const Sidebar = () => {
  const location = useLocation();

  return (
    <aside className="hidden md:flex fixed left-0 top-[104px] bottom-0 w-[64px] flex-col items-center py-6 gap-6 bg-[var(--bg-2)] border-r border-[var(--bg-3)] z-40 transition-all duration-300">
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
                ? 'text-[var(--accent-primary)] bg-[var(--bg-3)]' 
                : 'text-[var(--text-secondary)] hover:text-[var(--accent-primary)] hover:bg-[var(--bg-1)]'
            }`}
          >
            <Icon size={20} strokeWidth={1.5} />
            
            {/* Tooltip */}
            <div className="absolute left-full top-1/2 -translate-y-1/2 ml-4 px-3 py-1.5 rounded-lg opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity bg-[var(--bg-2)] border border-[var(--bg-3)] shadow-sm text-xs text-[var(--text-primary)] whitespace-nowrap z-50">
              {item.label}
            </div>
            
            {isActive && (
              <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-[var(--accent-primary)] rounded-r-full" />
            )}
          </Link>
        );
      })}
    </aside>
  );
};
