import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // three.js is split into the lazily loaded Model chunk (~585 kB); the rest stays small.
  build: { chunkSizeWarningLimit: 650 },
  // localhost only by default (`npm run dev -- --host` for a phone, with S2C_ACCESS_TOKEN set); xfwd tells the API
  // which device a relayed call came from, so its this-computer-only rule still holds through the proxy
  server: {
    proxy: { '/api': { target: 'http://127.0.0.1:8000', xfwd: true } },
  },
});
