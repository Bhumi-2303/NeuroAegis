import { Outlet } from 'react-router-dom';
import { TopNav } from './TopNav';
import { Sidebar } from './Sidebar';

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
