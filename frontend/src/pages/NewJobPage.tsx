import { type FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router";

import { useApi } from "../api/context";
import { useSubmit } from "../components/Form";
import { BLANK_JOB, JobFields, type JobForm, jobFrom } from "../components/JobFields";
import { ErrorMessage } from "../components/Status";

/** A new requisition starts open, taking applications, or as a draft that does not yet. */
const NEW_JOB_STATUSES = ["open", "draft"] as const;

export function NewJobPage() {
  const client = useApi();
  const navigate = useNavigate();
  const [form, setForm] = useState<JobForm>(BLANK_JOB);
  const { pending, error, run } = useSubmit();

  function set<K extends keyof JobForm>(key: K, value: JobForm[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const job = await run(() => client.createJob(jobFrom(form)));
    if (job) navigate(`/jobs/${job.id}`);
  }

  return (
    <section>
      <p>
        <Link to="/jobs">← All jobs</Link>
      </p>
      <h1>New job</h1>
      <form className="form" onSubmit={submit}>
        <JobFields
          form={form}
          set={set}
          statuses={NEW_JOB_STATUSES}
          statusHint="An open job takes applications; a draft does not yet."
        />
        {error && <ErrorMessage error={error} />}
        <div className="form-actions">
          <button type="submit" className="primary" disabled={pending}>
            {pending ? "Saving…" : "Create job"}
          </button>
          <Link to="/jobs">Cancel</Link>
        </div>
      </form>
    </section>
  );
}
