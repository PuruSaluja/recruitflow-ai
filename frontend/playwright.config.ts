import { defineConfig, devices } from "@playwright/test";

// End-to-end tests drive a real browser against a running API that serves the
// built client (`npm run build`, then uvicorn), with a trained matcher and a
// migrated database. They create their own job and candidates through the API
// and delete them afterwards.
//
// E2E_BASE_URL  where that server is (default http://127.0.0.1:8000)
// E2E_CHANNEL   an installed browser for the chromium project, e.g. msedge or
//               chrome, instead of Playwright's own download
const channel = process.env.E2E_CHANNEL || undefined;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  reporter: process.env.CI ? [["list"], ["github"]] : "list",
  timeout: 60_000,
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:8000",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"], channel } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
    { name: "mobile", use: { ...devices["Pixel 7"], channel } },
  ],
});
