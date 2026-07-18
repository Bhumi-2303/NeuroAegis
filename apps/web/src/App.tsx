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
          <RouterProvider router={router} />
        </StoreProvider>
      </QueryProvider>
    </ThemeProvider>
  );
}

export default App;
