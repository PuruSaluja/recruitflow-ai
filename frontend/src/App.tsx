import { NavLink, Navigate, Route, Routes } from "react-router";

import type { Health } from "./api/client";
import { useLoad } from "./api/context";
import { JobDetailPage } from "./pages/JobDetailPage";
import { JobsPage } from "./pages/JobsPage";
import { NotFound } from "./pages/NotFound";

function ApiStatus() {
  const [health] = useLoad<Health>("health", (api, signal) => api.health({ signal }));

  if (health.state === "loading") return <span className="muted">Checking the API…</span>;
  if (health.state === "error") return <span className="error">API: {health.error.message}</span>;
  return (
    <span className="muted">
      API {health.data.version}, database {health.data.database}
    </span>
  );
}

/** The pages by address; main.tsx puts them in a browser router, tests in a memory one. */
export function AppRoutes() {
  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">RecruitFlow AI</span>
        <nav aria-label="Main">
          <NavLink to="/jobs">Jobs</NavLink>
        </nav>
        <span className="spacer" />
        <ApiStatus />
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Navigate to="/jobs" replace />} />
          <Route path="/jobs" element={<JobsPage />} />
          <Route path="/jobs/:jobId" element={<JobDetailPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}
