import { Link, useLocation, Outlet } from 'react-router-dom';
import { LayoutDashboard, Brain, Activity } from 'lucide-react';
import { TopNav } from './TopNav';
import { Sidebar } from './Sidebar';

const MOBILE_NAV_ITEMS = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/analysis', icon: Brain, label: 'Analysis' },
  { path: '/eeg', icon: Activity, label: 'EEG' },
];

export const DashboardShell = () => {
  const location = useLocation();

  return (
    <div className="relative min-h-screen w-full flex flex-col bg-[var(--bg-1)] pb-16 md:pb-0">
      {/* Banner */}
      <div className="fixed top-0 left-0 right-0 z-[60] h-8 flex items-center justify-center bg-[var(--state-warning)] text-[var(--bg-1)] text-xs font-semibold uppercase tracking-wider">
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

      {/* Mobile Bottom Navigation */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 h-16 bg-[var(--bg-2)] border-t border-[var(--bg-4)] flex items-center justify-around px-2">
        {MOBILE_NAV_ITEMS.map((item) => {
          const isActive = location.pathname === item.path || 
                           (item.path !== '/' && location.pathname.startsWith(item.path));
          const Icon = item.icon;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex flex-col items-center gap-1 p-1 text-xs font-medium transition-colors ${
                isActive ? 'text-[var(--accent-primary)]' : 'text-[var(--text-secondary)]'
              }`}
            >
              <Icon size={20} strokeWidth={1.5} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
};
