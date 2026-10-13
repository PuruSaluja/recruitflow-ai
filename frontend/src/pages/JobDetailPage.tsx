import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import { ApiError, type Job, type MatchPage, type RankedCandidate } from "../api/client";
import { useApi, useLoad } from "../api/context";
import { Pager } from "../components/Pager";
import { SkillList } from "../components/SkillList";
import { ErrorMessage, Loaded } from "../components/Status";
import { date, label, percent, salaryBand, years } from "../format";
import { offsetFrom, parseId } from "../params";
import { NotFound } from "./NotFound";

export const MATCHES_PAGE_SIZE = 25;

function JobFacts({ job }: { job: Job }) {
  return (
    <>
      <header className="page-head">
        <div>
          <h1>{job.title}</h1>
          <p className="muted">
            {[job.department, job.location, label(job.employment_type)].filter(Boolean).join(" · ")}
          </p>
        </div>
        <span className={`badge status-${job.status}`}>{label(job.status)}</span>
      </header>

      {job.requirement_notes.length > 0 && (
        <div className="note" role="note">
          {job.requirement_notes.map((note) => (
            <p key={note}>{note}</p>
          ))}
        </div>
      )}

      <dl className="facts">
        <dt>Seniority</dt>
        <dd>{label(job.seniority)}</dd>
        <dt>Minimum experience</dt>
        <dd>{years(job.min_years_experience)}</dd>
        <dt>Education</dt>
        <dd>{label(job.education_required)}</dd>
        <dt>Salary</dt>
        <dd>{salaryBand(job.salary_min, job.salary_max)}</dd>
        <dt>Posted</dt>
        <dd>{date(job.created_at)}</dd>
      </dl>

      <h2>Required skills</h2>
      <SkillList skills={job.required_skills} />
      <h2>Preferred skills</h2>
      <SkillList skills={job.preferred_skills} />

      <h2>Description</h2>
      <p className="prose">{job.description}</p>
    </>
  );
}

function RankedTable({ jobId, rows }: { jobId: number; rows: RankedCandidate[] }) {
  const client = useApi();
  return (
    <table className="table">
      <thead>
        <tr>
          <th scope="col">Rank</th>
          <th scope="col">Candidate</th>
          <th scope="col">Score</th>
          <th scope="col">Required skills held</th>
          <th scope="col">Experience</th>
          <th scope="col">Seniority</th>
          <th scope="col">Scorecard</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.candidate.id}>
            <td>{row.rank}</td>
            <td>
              <Link to={`/candidates/${row.candidate.id}`}>{row.candidate.full_name}</Link>
              {row.candidate.current_title && (
                <div className="muted small">{row.candidate.current_title}</div>
              )}
            </td>
            <td className="score">
              <meter min={0} max={1} value={row.score} aria-hidden="true" /> {percent(row.score)}
            </td>
            <td>{percent(row.features.required_skill_coverage)}</td>
            <td>{years(row.candidate.years_experience)}</td>
            <td>{label(row.candidate.seniority)}</td>
            <td>
              <a
                href={client.scorecardPdfUrl(jobId, row.candidate.id)}
                aria-label={`Scorecard for ${row.candidate.full_name}`}
              >
                PDF
              </a>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Shortlist({ jobId }: { jobId: number }) {
  const client = useApi();
  const [params, setParams] = useSearchParams();
  const offset = offsetFrom(params);
  const [matches, reload] = useLoad<MatchPage>(`matches:${jobId}:${offset}`, (api, signal) =>
    api.listMatches(jobId, { limit: MATCHES_PAGE_SIZE, offset }, { signal }),
  );
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState<ApiError | null>(null);
  const [lastRun, setLastRun] = useState<{ count: number; ms: number } | null>(null);

  function goTo(next: number) {
    setParams((current) => {
      const updated = new URLSearchParams(current);
      if (next) updated.set("offset", String(next));
      else updated.delete("offset");
      return updated;
    });
  }

  async function score() {
    setScoring(true);
    setScoreError(null);
    try {
      // The whole pool is scored and cached either way; the ranking is read
      // back from the cache below, so only the run's totals are kept.
      const result = await client.scoreJob(jobId, { limit: 1 });
      setLastRun({ count: result.scored_count, ms: result.elapsed_ms });
      goTo(0);
      reload();
    } catch (error) {
      setScoreError(error instanceof ApiError ? error : new ApiError(0, "Scoring failed"));
    } finally {
      setScoring(false);
    }
  }

  return (
    <section aria-labelledby="shortlist-heading">
      <header className="section-head">
        <h2 id="shortlist-heading">Ranked candidates</h2>
        <div className="actions">
          {/* The API refuses a shortlist for a job with nothing scored (409). */}
          {matches.state === "ok" && matches.data.total > 0 && (
            <a href={client.shortlistPdfUrl(jobId)}>Shortlist PDF</a>
          )}
          <button type="button" onClick={score} disabled={scoring}>
            {scoring ? "Scoring…" : "Score all candidates"}
          </button>
        </div>
      </header>
      {lastRun && (
        <p className="muted" role="status">
          {lastRun.count === 0
            ? "There is no candidate on file to score."
            : `Scored ${lastRun.count} ${lastRun.count === 1 ? "candidate" : "candidates"} in ${Math.round(lastRun.ms)} ms.`}
        </p>
      )}
      {scoreError && <ErrorMessage error={scoreError} />}

      <Loaded value={matches} what="the ranking" retry={reload}>
        {(page) =>
          page.total === 0 ? (
            <p className="muted">
              No candidate has been scored for this job by the current matcher yet. Score them to
              build the ranking.
            </p>
          ) : (
            <>
              <p className="muted small">
                A score is the matcher&apos;s estimate that a recruiter would advance the candidate,
                as of when it was scored. Matcher {page.model_version.slice(0, 12)}.
              </p>
              {page.items.length === 0 ? (
                <p className="muted">No candidates on this page.</p>
              ) : (
                <RankedTable jobId={jobId} rows={page.items} />
              )}
              <Pager total={page.total} limit={MATCHES_PAGE_SIZE} offset={offset} onChange={goTo} />
            </>
          )
        }
      </Loaded>
    </section>
  );
}

export function JobDetailPage() {
  const jobId = parseId(useParams().jobId);
  const [job, reload] = useLoad<Job | null>(`job:${jobId}`, (api, signal) =>
    jobId === null ? Promise.resolve(null) : api.getJob(jobId, { signal }),
  );

  if (jobId === null) return <NotFound what="job" />;
  if (job.state === "error" && job.error.status === 404) return <NotFound what="job" />;

  return (
    <article>
      <p>
        <Link to="/jobs">← All jobs</Link>
      </p>
      <Loaded value={job} what="the job" retry={reload}>
        {(data) =>
          data && (
            <>
              <JobFacts job={data} />
              <Shortlist jobId={data.id} />
            </>
          )
        }
      </Loaded>
    </article>
  );
}
