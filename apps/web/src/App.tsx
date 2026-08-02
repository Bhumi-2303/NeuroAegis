import { RouterProvider } from 'react-router-dom';
import { QueryProvider, StoreProvider, ThemeProvider } from './app/providers';
import { router } from './app/routes';
import { ParticleField } from './design-system/primitives';

function App() {
  return (
    <ThemeProvider>
      <QueryProvider>
        <StoreProvider>
          <ParticleField count={40} />
          
          <div style={{
            backgroundColor: '#ff9800',
            color: '#fff',
            padding: '8px 16px',
            textAlign: 'center',
            fontWeight: 'bold',
            position: 'relative',
            zIndex: 1000
          }}>
            ⚠️ Not for clinical use — research prototype
          </div>

          <RouterProvider router={router} />
        </StoreProvider>
      </QueryProvider>
    </ThemeProvider>
  );
}

export default App;
