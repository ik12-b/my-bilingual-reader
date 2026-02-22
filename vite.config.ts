import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import type { Plugin } from 'vite';

// Plugin: return 404 for /custom-model/* so Transformers.js gets proper not-found responses
// Without this, Vite's SPA fallback serves index.html which crashes JSON parsing.
function customModelNotFoundPlugin(): Plugin {
  return {
    name: 'custom-model-404',
    configureServer(server) {
      // Use 'use' at the very beginning of the middleware stack
      // by inserting before other middleware
      server.middlewares.use((req, res, next) => {
        const rawUrl = req.url || '';
        const url = rawUrl.split('?')[0];

        // Log ALL requests containing 'custom-model' to find any that bypass our check
        if (rawUrl.includes('custom-model')) {
          console.log(`[Vite Middleware] Request: ${rawUrl}`);
        }

        // Intercept /custom-model, /custom-model/, /custom-model/...
        if (url === '/custom-model' || url.startsWith('/custom-model/')) {
          console.log(`[Vite Middleware] Returning 404 for: ${url}`);
          res.writeHead(404, {
            'Content-Type': 'text/plain',
            'Cache-Control': 'no-store'
          });
          res.end(`404 Not Found: ${url}`);
          return;
        }
        next();
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');

  return {
    plugins: [react(), tailwindcss(), customModelNotFoundPlugin()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify: file watching is disabled to prevent flickering during agent edits.
      hmr: env.DISABLE_HMR !== 'true',
    },
  };
});

