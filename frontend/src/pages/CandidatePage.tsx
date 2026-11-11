import { type ChangeEvent, type FormEvent, useState } from "react";
import { Link, useParams } from "react-router";

import type { Application, ApplicationStatus, Candidate, Job } from "../api/client";
import { useApi, useLoad } from "../api/context";
import { Field, optional, useSubmit } from "../components/Form";
import { SkillList } from "../components/SkillList";
import { ErrorMessage, Loaded } from "../components/Status";
import { date, label, years } from "../format";
import { parseId } from "../params";
import { RESUME_TYPES } from "./NewCandidatePage";
import { NotFound } from "./NotFound";

/** Most applications listed for one candidate; the API's largest page. */
export const APPLICATIONS_SHOWN = 200;

/** Most open jobs offered to apply to; the API's largest page. */
export const OPEN_JOBS_SHOWN = 200;

type ApplicationRow = { application: Application; job: Job | null };

function jobTitle({ application, job }: ApplicationRow): string {
  return job ? job.title : `Job ${application.job_id}`;
}

/**
 * An application's stage, with the moves the API lists for it: the next
 * stages, or a reopen for one rejected or withdrawn.
 */
function StageControls({ row, onChanged }: { row: ApplicationRow; onChanged: () => void }) {
  const client = useApi();
  const { application } = row;
  const title = jobTitle(row);
  const [target, setTarget] = useState<ApplicationStatus | "">("");
  const { pending, error, run } = useSubmit();

  async function move(event: FormEvent) {
    event.preventDefault();
    if (!target) return;
    const moved = await run(() => client.updateApplication(application.id, { status: target }));
    if (moved) {
      setTarget("");
      onChanged();
    }
  }

  async function reopen() {
    if (await run(() => client.reopenApplication(application.id))) onChanged();
  }

  return (
    <div className="stage">
      <span>{label(application.status)}</span>
      {application.next_statuses.length > 0 && (
        <form className="inline-form" onSubmit={move}>
          <select
            aria-label={`Next stage for ${title}`}
            value={target}
            onChange={(event) => setTarget(event.target.value as ApplicationStatus | "")}
            disabled={pending}
          >
            <option value="">Move to…</option>
            {application.next_statuses.map((status) => (
              <option key={status} value={status}>
                {label(status)}
              </option>
            ))}
          </select>
          <button type="submit" disabled={pending || !target} aria-label={`Move the application for ${title}`}>
            Move
          </button>
        </form>
      )}
      {application.reopenable && (
        <button
          type="button"
          onClick={reopen}
          disabled={pending}
          aria-label={`Reopen the application for ${title}`}
        >
          Reopen
        </button>
      )}
      {error && <ErrorMessage error={error} />}
    </div>
  );
}

/** Applying to an open job. Shown only once asked for, so its jobs are read on demand. */
function ApplyForm({
  candidateId,
  appliedTo,
  onApplied,
}: {
  candidateId: number;
  appliedTo: Set<number>;
  onApplied: () => void;
}) {
  const client = useApi();
  const [jobs, reload] = useLoad(`open-jobs:${candidateId}`, (api, signal) =>
    api.listJobs({ status: "open", limit: OPEN_JOBS_SHOWN }, { signal }),
  );
  const [jobId, setJobId] = useState("");
  const [source, setSource] = useState("");
  const { pending, error, run } = useSubmit();

  async function apply(event: FormEvent) {
    event.preventDefault();
    const applied = await run(() =>
      client.createApplication({ job_id: Number(jobId), candidate_id: candidateId, source: optional(source) }),
    );
    if (applied) onApplied();
  }

  return (
    <Loaded value={jobs} what="open jobs" retry={reload}>
      {(page) => {
        const open = page.items.filter((job) => !appliedTo.has(job.id));
        if (open.length === 0) {
          return <p className="muted">There is no open job this candidate has not applied to.</p>;
        }
        return (
          <form className="form" onSubmit={apply}>
            <div className="form-row">
              <Field label="Job">
                {(props) => (
                  <select {...props} required value={jobId} onChange={(event) => setJobId(event.target.value)}>
                    <option value="">Choose an open job</option>
                    {open.map((job) => (
                      <option key={job.id} value={job.id}>
                        {job.title}
                      </option>
                    ))}
                  </select>
                )}
              </Field>
              <Field label="Source" hint="Where the application came from, such as a referral.">
                {(props) => (
                  <input {...props} maxLength={80} value={source} onChange={(event) => setSource(event.target.value)} />
                )}
              </Field>
            </div>
            {page.total > page.items.length && (
              <p className="muted small">
                Offering the newest {page.items.length} of {page.total} open jobs.
              </p>
            )}
            {error && <ErrorMessage error={error} />}
            <div className="form-actions">
              <button type="submit" className="primary" disabled={pending || !jobId}>
                {pending ? "Applying…" : "Apply"}
              </button>
            </div>
          </form>
        );
      }}
    </Loaded>
  );
}

function Applications({ candidateId }: { candidateId: number }) {
  const [rows, reload] = useLoad<{ rows: ApplicationRow[]; total: number }>(
    `applications:${candidateId}`,
    async (api, signal) => {
      const page = await api.listApplications(
        { candidate_id: candidateId, limit: APPLICATIONS_SHOWN },
        { signal },
      );
      // The jobs' titles in one read, however many applications there are. A job
      // deleted in between, or a failed read, leaves its application shown by job id.
      const ids = [...new Set(page.items.map((application) => application.job_id))];
      const jobs =
        ids.length === 0
          ? null
          : await api.listJobs({ id: ids, limit: APPLICATIONS_SHOWN }, { signal }).catch(() => null);
      const byId = new Map((jobs?.items ?? []).map((job) => [job.id, job]));
      return {
        rows: page.items.map((application) => ({
          application,
          job: byId.get(application.job_id) ?? null,
        })),
        total: page.total,
      };
    },
  );
  const [applying, setApplying] = useState(false);
  const appliedTo = new Set(rows.state === "ok" ? rows.data.rows.map((row) => row.application.job_id) : []);

  return (
    <section aria-labelledby="applications-heading">
      <header className="section-head">
        <h2 id="applications-heading">Applications</h2>
        <div className="actions">
          <button type="button" aria-expanded={applying} onClick={() => setApplying((open) => !open)}>
            {applying ? "Close" : "Apply to a job"}
          </button>
        </div>
      </header>
      {applying && rows.state === "ok" && (
        <ApplyForm
          candidateId={candidateId}
          appliedTo={appliedTo}
          onApplied={() => {
            setApplying(false);
            reload();
          }}
        />
      )}
      <Loaded value={rows} what="applications" retry={reload}>
        {({ rows: items, total }) =>
          items.length === 0 ? (
            <p className="muted">No applications on file.</p>
          ) : (
            <>
              <table className="table">
                <thead>
                  <tr>
                    <th scope="col">Job</th>
                    <th scope="col">Stage</th>
                    <th scope="col">Source</th>
                    <th scope="col">Applied</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((row) => (
                    <tr key={row.application.id}>
                      <td>
                        <Link to={`/jobs/${row.application.job_id}`}>{jobTitle(row)}</Link>
                      </td>
                      <td>
                        <StageControls row={row} onChanged={reload} />
                      </td>
                      <td>{row.application.source ?? "—"}</td>
                      <td>{date(row.application.applied_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {total > items.length && (
                <p className="muted small">
                  Showing the latest {items.length} of {total}.
                </p>
              )}
            </>
          )
        }
      </Loaded>
    </section>
  );
}

/** Replacing the stored resume with a file; the API fills an empty skill list from it. */
function ResumeUpload({ candidateId, onUploaded }: { candidateId: number; onUploaded: () => void }) {
  const client = useApi();
  const { pending, error, run } = useSubmit();

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    // Cleared at once, so choosing the same file again uploads it again.
    event.target.value = "";
    if (file && (await run(() => client.uploadResume(candidateId, file)))) onUploaded();
  }

  return (
    <div className="upload">
      <Field label="Replace with a resume file" hint="PDF, plain text or Markdown, up to 2 MB.">
        {(props) => <input {...props} type="file" accept={RESUME_TYPES} onChange={upload} disabled={pending} />}
      </Field>
      {pending && (
        <p className="muted" role="status">
          Uploading…
        </p>
      )}
      {error && <ErrorMessage error={error} />}
    </div>
  );
}

function CandidateFacts({ candidate, onChanged }: { candidate: Candidate; onChanged: () => void }) {
  return (
    <>
      <header className="page-head">
        <div>
          <h1>{candidate.full_name}</h1>
          <p className="muted">
            {[candidate.current_title, candidate.location].filter(Boolean).join(" · ") || "—"}
          </p>
        </div>
      </header>

      <dl className="facts">
        <dt>Email</dt>
        <dd>
          <a href={`mailto:${candidate.email}`}>{candidate.email}</a>
        </dd>
        {candidate.phone && (
          <>
            <dt>Phone</dt>
            <dd>{candidate.phone}</dd>
          </>
        )}
        <dt>Experience</dt>
        <dd>{years(candidate.years_experience)}</dd>
        <dt>Seniority</dt>
        <dd>{label(candidate.seniority)}</dd>
        <dt>Education</dt>
        <dd>{label(candidate.education_level)}</dd>
        <dt>On file since</dt>
        <dd>{date(candidate.created_at)}</dd>
      </dl>

      <h2>Skills</h2>
      <SkillList skills={candidate.skills} />

      <h2>Resume</h2>
      <pre className="resume">{candidate.resume_text}</pre>
      <ResumeUpload candidateId={candidate.id} onUploaded={onChanged} />
    </>
  );
}

export function CandidatePage() {
  const candidateId = parseId(useParams().candidateId);
  const [candidate, reload] = useLoad<Candidate | null>(`candidate:${candidateId}`, (api, signal) =>
    candidateId === null ? Promise.resolve(null) : api.getCandidate(candidateId, { signal }),
  );

  if (candidateId === null) return <NotFound what="candidate" />;
  if (candidate.state === "error" && candidate.error.status === 404) {
    return <NotFound what="candidate" />;
  }

  return (
    <article>
      <Loaded value={candidate} what="the candidate" retry={reload}>
        {(data) =>
          data && (
            <>
              <CandidateFacts candidate={data} onChanged={reload} />
              <Applications candidateId={data.id} />
            </>
          )
        }
      </Loaded>
    </article>
  );
}
