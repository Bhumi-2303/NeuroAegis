import { Link, useLocation } from 'react-router-dom';
import { Search, Bell, User, Brain } from 'lucide-react';

const TOP_LINKS = [
  { path: '/', label: 'Dashboard' },
  { path: '/analysis', label: 'Analysis' },
  { path: '/reports', label: 'Reports' },
];

export const TopNav = () => {
  const location = useLocation();

  return (
    <header className="fixed top-8 left-0 right-0 z-50 h-[72px] w-full px-6 flex items-center justify-between bg-[var(--bg-2)] border-b border-[var(--bg-3)]">
      {/* Left — Brand */}
      <div className="flex items-center gap-3">
        <div className="text-[var(--accent-primary)]">
          <Brain size={28} strokeWidth={1.5} />
        </div>
        <span className="font-display font-semibold text-lg tracking-wide text-[var(--text-primary)]">
          NeuroAegis
        </span>
      </div>

      {/* Center — Primary nav links */}
      <nav className="hidden md:flex items-center gap-8">
        {TOP_LINKS.map((link) => {
          const isActive = location.pathname === link.path ||
            (link.path !== '/' && location.pathname.startsWith(link.path));
          return (
            <Link
              key={link.path}
              to={link.path}
              className={`text-sm font-medium transition-colors ${
                isActive
                  ? 'text-[var(--accent-primary)] border-b-2 border-[var(--accent-primary)] pb-1 font-semibold'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>

      {/* Right — Actions */}
      <div className="flex items-center gap-5">
        <button className="flex items-center gap-2 text-[var(--text-secondary)] hover:text-[var(--accent-primary)] transition-colors group">
          <Search size={20} strokeWidth={1.5} />
          <span className="hidden lg:inline-flex border border-[var(--bg-3)] rounded px-1.5 py-0.5 text-xs bg-[var(--bg-1)] group-hover:border-[var(--accent-primary)] transition-colors">
            ⌘K
          </span>
        </button>
        <button className="relative text-[var(--text-secondary)] hover:text-[var(--accent-primary)] transition-colors">
          <Bell size={20} strokeWidth={1.5} />
          <span className="absolute top-0 right-0 w-2 h-2 bg-[var(--state-danger)] rounded-full border border-[var(--bg-2)]"></span>
        </button>
        <button className="w-8 h-8 rounded-full bg-[var(--bg-1)] border border-[var(--bg-3)] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--accent-primary)] hover:border-[var(--accent-primary)] transition-colors">
          <User size={16} strokeWidth={1.5} />
        </button>
      </div>
    </header>
  );
};
