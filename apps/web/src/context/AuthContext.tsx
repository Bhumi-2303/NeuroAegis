import React, { createContext, useContext, useEffect, useState, useCallback, ReactNode } from 'react';
import {
  AuthUser,
  loginUser,
  getMe,
  logoutUser,
  logoutAllSessions,
  refreshAuthSession,
  getCsrfToken,
  onSessionExpired,
  ApiRequestError,
} from '../services/api';

export type AuthStatus = 'loading' | 'authenticated' | 'unauthenticated';

export interface AuthContextType {
  user: AuthUser | null;
  status: AuthStatus;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  logoutAll: () => Promise<void>;
  refresh: () => Promise<boolean>;
  clearError: () => void;
  isAdmin: boolean;
  isClinician: boolean;
  isResearcher: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [status, setStatus] = useState<AuthStatus>('loading');
  const [error, setError] = useState<string | null>(null);

  const clearError = useCallback(() => setError(null), []);

  const bootstrapSession = useCallback(async () => {
    setStatus('loading');
    setError(null);
    try {
      const currentUser = await getMe();
      setUser(currentUser);
      setStatus('authenticated');
    } catch (err) {
      if (err instanceof ApiRequestError && (err.status === 401 || err.status === 403)) {
        // Only try refresh if an active session cookie was present
        if (getCsrfToken()) {
          const refreshed = await refreshAuthSession();
          if (refreshed) {
            try {
              const reloadedUser = await getMe();
              setUser(reloadedUser);
              setStatus('authenticated');
              return;
            } catch {}
          }
        }
      }
      setUser(null);
      setStatus('unauthenticated');
    }
  }, []);

  useEffect(() => {
    bootstrapSession();

    // Subscribe to session expiration events triggered by apiFetch
    const unsubscribe = onSessionExpired(() => {
      setUser(null);
      setStatus('unauthenticated');
    });

    return () => {
      unsubscribe();
    };
  }, [bootstrapSession]);

  const login = useCallback(async (username: string, password: string) => {
    setError(null);
    try {
      const loggedInUser = await loginUser(username, password);
      setUser(loggedInUser);
      setStatus('authenticated');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Login failed';
      setError(msg);
      throw err;
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await logoutUser();
    } finally {
      setUser(null);
      setStatus('unauthenticated');
      setError(null);
    }
  }, []);

  const logoutAll = useCallback(async () => {
    try {
      await logoutAllSessions();
    } finally {
      setUser(null);
      setStatus('unauthenticated');
      setError(null);
    }
  }, []);

  const refresh = useCallback(async (): Promise<boolean> => {
    const success = await refreshAuthSession();
    if (success) {
      try {
        const refreshedUser = await getMe();
        setUser(refreshedUser);
        setStatus('authenticated');
        return true;
      } catch {
        setUser(null);
        setStatus('unauthenticated');
        return false;
      }
    } else {
      setUser(null);
      setStatus('unauthenticated');
      return false;
    }
  }, []);

  const isAdmin = user?.role === 'admin';
  const isClinician = user?.role === 'clinician';
  const isResearcher = user?.role === 'researcher';

  return (
    <AuthContext.Provider
      value={{
        user,
        status,
        error,
        login,
        logout,
        logoutAll,
        refresh,
        clearError,
        isAdmin,
        isClinician,
        isResearcher,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
