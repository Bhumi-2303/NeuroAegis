import type { ReactNode } from 'react';
import { useEffect } from 'react';

export const ThemeProvider = ({ children }: { children: ReactNode }) => {
  useEffect(() => {
    // Ensuring the root element has the dark class if needed,
    // though our entire app is dark-themed by default.
    document.documentElement.classList.add('dark');
  }, []);

  return <>{children}</>;
};
