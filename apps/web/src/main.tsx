import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

const originalFetch = window.fetch;
window.fetch = async (...args) => {
  const [resource, config] = args;
  const token = localStorage.getItem('token');
  
  if (token && typeof resource === 'string' && resource.includes('/api/')) {
    const newConfig = {
      ...config,
      headers: {
        ...(config?.headers || {}),
        'Authorization': `Bearer ${token}`
      }
    };
    return originalFetch(resource, newConfig);
  }
  return originalFetch(...args);
};

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
