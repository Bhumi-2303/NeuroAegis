import type { ReactNode } from 'react';

// For now, this is a passthrough since Zustand is global by default.
// It exists to fulfill architectural constraints if we ever need to
// inject a specific store instance via React Context.
export const StoreProvider = ({ children }: { children: ReactNode }) => {
  return <>{children}</>;
};
