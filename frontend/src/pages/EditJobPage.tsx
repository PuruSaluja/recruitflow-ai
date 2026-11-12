import { type FormEvent, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router";

import type { Job } from "../api/client";
import { useApi, useLoad } from "../api/context";
import { changed, useSubmit } from "../components/Form";
import { formFromJob, jobAsChange, JobFields, type JobForm } from "../components/JobFields";
import { ErrorMessage, Loaded } from "../components/Status";
import { JOB_STATUSES } from "../options";
import { parseId } from "../params";
import { NotFound } from "./NotFound";

/**
 * The fields the matcher reads. Scores are kept as they were made, so a change
 * to one of these leaves a ranking that describes the job as it was.
 */
const SCORED_FIELDS = [
  "title",
  "description",
  "seniority",
  "education_required",
  "min_years_experience",
  "required_skills",
  "preferred_skills",
] as const;

/**
 * What the job's page is told on the way back from a change the matcher reads.
 * It comes into the form with the reader and goes back with them, so it lasts
 * through a later change, or none, until the candidates are scored again.
 */
export type AfterEdit = { rescore: true };

function EditJobForm({ job }: { job: Job }) {
  const client = useApi();
  const navigate = useNavigate();
  const [form, setForm] = useState<JobForm>(() => formFromJob(job));
  const { pending, error, run } = useSubmit();
  const back = `/jobs/${job.id}`;
  const before = useLocation().state as AfterEdit | null;

  function set<K extends keyof JobForm>(key: K, value: JobForm[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const changes = changed(jobAsChange(formFromJob(job)), jobAsChange(form));
    // With nothing changed there is nothing to ask of the API.
    if (Object.keys(changes).length === 0) {
      navigate(back, { state: before });
    } else if (await run(() => client.updateJob(job.id, changes))) {
      const state: AfterEdit | null = SCORED_FIELDS.some((field) => field in changes) ? { rescore: true } : before;
      navigate(back, { state });
    }
  }

  return (
    <section>
      <p>
        <Link to={back} state={before}>
          ← {job.title}
        </Link>
      </p>
      <h1>Edit job</h1>
      <form className="form" onSubmit={submit}>
        <JobFields
          form={form}
          set={set}
          statuses={JOB_STATUSES}
          statusHint="Only an open job takes applications."
        />
        {error && <ErrorMessage error={error} />}
        <div className="form-actions">
          <button type="submit" className="primary" disabled={pending}>
            {pending ? "Saving…" : "Save changes"}
          </button>
          <Link to={back} state={before}>
            Cancel
          </Link>
        </div>
      </form>
    </section>
  );
}

export function EditJobPage() {
  const jobId = parseId(useParams().jobId);
  const [job, reload] = useLoad<Job | null>(`job:${jobId}`, (api, signal) =>
    jobId === null ? Promise.resolve(null) : api.getJob(jobId, { signal }),
  );

  if (jobId === null) return <NotFound what="job" />;
  if (job.state === "error" && job.error.status === 404) return <NotFound what="job" />;

  return (
    <Loaded value={job} what="the job" retry={reload}>
      {(data) => data && <EditJobForm job={data} />}
    </Loaded>
  );
}
