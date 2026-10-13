import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The dev server forwards /api to the FastAPI app, so the browser sees one
// origin and the API needs no CORS entry for it. RECRUITFLOW_API_TARGET points
// it at an API running somewhere other than uvicorn's default address.
const apiTarget = process.env.RECRUITFLOW_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    // e2e/ holds the Playwright tests, run against a live server by `npm run e2e`.
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["./src/test/setup.ts"],
  },
});
