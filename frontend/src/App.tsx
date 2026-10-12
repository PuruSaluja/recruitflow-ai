import { useEffect, useState } from "react";

import { api, ApiError, type Health } from "./api/client";

type Status = { state: "loading" } | { state: "ok"; health: Health } | { state: "error"; message: string };

function ApiStatus() {
  const [status, setStatus] = useState<Status>({ state: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    api
      .health({ signal: controller.signal })
      .then((health) => setStatus({ state: "ok", health }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setStatus({
          state: "error",
          message: error instanceof ApiError ? error.message : "Unexpected error",
        });
      });
    return () => controller.abort();
  }, []);

  if (status.state === "loading") return <span className="muted">Checking the API…</span>;
  if (status.state === "error") return <span className="error">API: {status.message}</span>;
  return (
    <span className="muted">
      API {status.health.version}, database {status.health.database}
    </span>
  );
}

export function App() {
  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">RecruitFlow AI</span>
        <ApiStatus />
      </header>
      <main className="content">
        <p>Jobs, candidates and ranked shortlists.</p>
      </main>
    </div>
  );
}
