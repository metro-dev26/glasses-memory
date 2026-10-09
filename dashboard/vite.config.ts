import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Two pages: the dashboard at / and the camera page at /phone/.
// In `npm run dev`, websockets and images go to the mock server on :8000.
const server = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        dashboard: resolve(__dirname, "index.html"),
        phone: resolve(__dirname, "phone/index.html"),
      },
    },
  },
  server: {
    host: true,
    proxy: {
      "/ws": { target: server, ws: true },
      "/thumb": server,
      "/snapshot": server,
      "/keyframe": server,
    },
  },
});
