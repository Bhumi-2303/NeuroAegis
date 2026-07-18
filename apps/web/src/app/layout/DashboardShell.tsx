import { Outlet } from 'react-router-dom';
import { TopNav } from './TopNav';
import { Sidebar } from './Sidebar';

export const DashboardShell = () => {
  return (
    <div className="relative min-h-screen w-full flex flex-col">
      <TopNav />
      <div className="flex flex-1 pt-[72px]">
        <Sidebar />
        <main className="flex-1 ml-0 md:ml-[64px] p-4 md:p-6 min-w-0 transition-all duration-300">
          <Outlet />
        </main>
      </div>
    </div>
  );
};
