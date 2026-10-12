import { type FormEvent, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router";

import type { JobQuery } from "../api/client";
import { useLoad } from "../api/context";
import { Pager } from "../components/Pager";
import { Loaded } from "../components/Status";
import { date, label } from "../format";
import { offsetFrom } from "../params";

export const JOBS_PAGE_SIZE = 20;
const STATUSES = ["open", "paused", "draft", "closed"] as const;

export function JobsPage() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") ?? "";
  const statusParam = params.get("status");
  const status = STATUSES.find((s) => s === statusParam);
  const offset = offsetFrom(params);
  const [search, setSearch] = useState(q);
  // Back and forward change `q` without a keystroke; keep the box in step.
  useEffect(() => setSearch(q), [q]);

  const query: JobQuery = { q: q || undefined, status, limit: JOBS_PAGE_SIZE, offset };
  const [jobs, reload] = useLoad(`jobs:${q}:${status ?? ""}:${offset}`, (client, signal) =>
    client.listJobs(query, { signal }),
  );

  function update(changes: Record<string, string | undefined>) {
    const next = new URLSearchParams(params);
    for (const [key, value] of Object.entries(changes)) {
      if (value) next.set(key, value);
      else next.delete(key);
    }
    setParams(next);
  }

  function onSearch(event: FormEvent) {
    event.preventDefault();
    update({ q: search.trim() || undefined, offset: undefined });
  }

  return (
    <section>
      <h1>Jobs</h1>
      <form className="filters" onSubmit={onSearch} role="search">
        <input
          type="search"
          aria-label="Search job titles"
          placeholder="Search titles"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <select
          aria-label="Status"
          value={status ?? ""}
          onChange={(event) => update({ status: event.target.value || undefined, offset: undefined })}
        >
          <option value="">Any status</option>
          {STATUSES.map((s) => (
            <option key={s} value={s}>
              {label(s)}
            </option>
          ))}
        </select>
        <button type="submit">Search</button>
      </form>

      <Loaded value={jobs} what="jobs" retry={reload}>
        {(page) =>
          page.items.length === 0 ? (
            <p className="muted">{offset > 0 ? "No jobs on this page." : "No jobs match."}</p>
          ) : (
            <>
              <table className="table">
                <thead>
                  <tr>
                    <th scope="col">Title</th>
                    <th scope="col">Department</th>
                    <th scope="col">Location</th>
                    <th scope="col">Seniority</th>
                    <th scope="col">Status</th>
                    <th scope="col">Posted</th>
                  </tr>
                </thead>
                <tbody>
                  {page.items.map((job) => (
                    <tr key={job.id}>
                      <td>
                        <Link to={`/jobs/${job.id}`}>{job.title}</Link>
                      </td>
                      <td>{job.department ?? "—"}</td>
                      <td>{job.location ?? "—"}</td>
                      <td>{label(job.seniority)}</td>
                      <td>
                        <span className={`badge status-${job.status}`}>{label(job.status)}</span>
                      </td>
                      <td>{date(job.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <Pager
                total={page.total}
                limit={JOBS_PAGE_SIZE}
                offset={offset}
                onChange={(next) => update({ offset: next ? String(next) : undefined })}
              />
            </>
          )
        }
      </Loaded>
    </section>
  );
}
