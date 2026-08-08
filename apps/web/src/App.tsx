import { RouterProvider } from 'react-router-dom';
import { QueryProvider, StoreProvider, ThemeProvider } from './app/providers';
import { router } from './app/routes';
import { ParticleField } from './design-system/primitives';

import { AuthProvider } from './app/providers/AuthProvider';

function App() {
  return (
    <AuthProvider>
      <ThemeProvider>
        <QueryProvider>
          <StoreProvider>
            <ParticleField count={40} />

            <RouterProvider router={router} />
          </StoreProvider>
        </QueryProvider>
      </ThemeProvider>
    </AuthProvider>
  );
}

export default App;
