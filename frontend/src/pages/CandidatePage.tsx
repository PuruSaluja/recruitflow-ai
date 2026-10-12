import { Link, useParams } from "react-router";

import type { Application, Candidate, Job } from "../api/client";
import { useLoad } from "../api/context";
import { SkillList } from "../components/SkillList";
import { Loaded } from "../components/Status";
import { date, label, years } from "../format";
import { parseId } from "../params";
import { NotFound } from "./NotFound";

/** Most applications listed for one candidate; the API's largest page. */
export const APPLICATIONS_SHOWN = 200;

type ApplicationRow = { application: Application; job: Job | null };

function Applications({ candidateId }: { candidateId: number }) {
  const [rows, reload] = useLoad<{ rows: ApplicationRow[]; total: number }>(
    `applications:${candidateId}`,
    async (api, signal) => {
      const page = await api.listApplications(
        { candidate_id: candidateId, limit: APPLICATIONS_SHOWN },
        { signal },
      );
      // One read per distinct job, for its title; a job deleted in between is shown by id.
      const ids = [...new Set(page.items.map((application) => application.job_id))];
      const jobs = await Promise.all(
        ids.map((id) => api.getJob(id, { signal }).catch(() => null)),
      );
      const byId = new Map(ids.map((id, index) => [id, jobs[index] ?? null]));
      return {
        rows: page.items.map((application) => ({
          application,
          job: byId.get(application.job_id) ?? null,
        })),
        total: page.total,
      };
    },
  );

  return (
    <section aria-labelledby="applications-heading">
      <h2 id="applications-heading">Applications</h2>
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
                  {items.map(({ application, job }) => (
                    <tr key={application.id}>
                      <td>
                        <Link to={`/jobs/${application.job_id}`}>
                          {job ? job.title : `Job ${application.job_id}`}
                        </Link>
                      </td>
                      <td>{label(application.status)}</td>
                      <td>{application.source ?? "—"}</td>
                      <td>{date(application.applied_at)}</td>
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

function CandidateFacts({ candidate }: { candidate: Candidate }) {
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
              <CandidateFacts candidate={data} />
              <Applications candidateId={data.id} />
            </>
          )
        }
      </Loaded>
    </article>
  );
}
