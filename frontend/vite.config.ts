import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server is loopback-only. /api, /streams, /ws -> backend (8000); /sim -> simulator (8001).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/streams": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
      "/sim": { target: "http://127.0.0.1:8001", rewrite: (p) => p.replace(/^\/sim/, "") },
    },
  },
});
