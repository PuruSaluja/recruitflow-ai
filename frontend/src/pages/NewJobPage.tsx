import { type FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router";

import type { NewJob } from "../api/client";
import { useApi } from "../api/context";
import { Field, listFrom, optional, useSubmit } from "../components/Form";
import { ErrorMessage } from "../components/Status";
import { label } from "../format";
import { EDUCATION_LEVELS, EMPLOYMENT_TYPES, SENIORITY_LEVELS } from "../options";
/** A new requisition starts open, taking applications, or as a draft that does not yet. */
const NEW_JOB_STATUSES = ["open", "draft"] as const;

type Form = {
  title: string;
  department: string;
  location: string;
  employment_type: (typeof EMPLOYMENT_TYPES)[number];
  seniority: (typeof SENIORITY_LEVELS)[number];
  education_required: (typeof EDUCATION_LEVELS)[number];
  min_years_experience: string;
  required_skills: string;
  preferred_skills: string;
  salary_min: string;
  salary_max: string;
  status: (typeof NEW_JOB_STATUSES)[number];
  description: string;
};

/** The API's own defaults, so leaving a field alone sends what the API would assume. */
const BLANK: Form = {
  title: "",
  department: "",
  location: "",
  employment_type: "full_time",
  seniority: "mid",
  education_required: "bachelor",
  min_years_experience: "0",
  required_skills: "",
  preferred_skills: "",
  salary_min: "",
  salary_max: "",
  status: "open",
  description: "",
};

/** The request body: blanks left out, so the API applies its defaults and checks. */
export function jobFrom(form: Form): NewJob {
  return {
    title: form.title,
    description: form.description,
    department: optional(form.department),
    location: optional(form.location),
    employment_type: form.employment_type,
    seniority: form.seniority,
    education_required: form.education_required,
    min_years_experience: Number(form.min_years_experience || 0),
    required_skills: listFrom(form.required_skills),
    preferred_skills: listFrom(form.preferred_skills),
    // Salaries go as the typed text: the API reads decimals exactly, where a
    // float could not hold every amount of cents.
    salary_min: optional(form.salary_min),
    salary_max: optional(form.salary_max),
    status: form.status,
  };
}

export function NewJobPage() {
  const client = useApi();
  const navigate = useNavigate();
  const [form, setForm] = useState<Form>(BLANK);
  const { pending, error, run } = useSubmit();

  function set<K extends keyof Form>(key: K, value: Form[K]) {
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
        <Field label="Title">
          {(props) => (
            <input
              {...props}
              required
              minLength={2}
              maxLength={200}
              value={form.title}
              onChange={(event) => set("title", event.target.value)}
            />
          )}
        </Field>
        <div className="form-row">
          <Field label="Department">
            {(props) => (
              <input
                {...props}
                maxLength={100}
                value={form.department}
                onChange={(event) => set("department", event.target.value)}
              />
            )}
          </Field>
          <Field label="Location">
            {(props) => (
              <input
                {...props}
                maxLength={150}
                value={form.location}
                onChange={(event) => set("location", event.target.value)}
              />
            )}
          </Field>
        </div>
        <div className="form-row">
          <Field label="Employment type">
            {(props) => (
              <select
                {...props}
                value={form.employment_type}
                onChange={(event) => set("employment_type", event.target.value as Form["employment_type"])}
              >
                {EMPLOYMENT_TYPES.map((value) => (
                  <option key={value} value={value}>
                    {label(value)}
                  </option>
                ))}
              </select>
            )}
          </Field>
          <Field label="Seniority">
            {(props) => (
              <select
                {...props}
                value={form.seniority}
                onChange={(event) => set("seniority", event.target.value as Form["seniority"])}
              >
                {SENIORITY_LEVELS.map((value) => (
                  <option key={value} value={value}>
                    {label(value)}
                  </option>
                ))}
              </select>
            )}
          </Field>
          <Field label="Education required">
            {(props) => (
              <select
                {...props}
                value={form.education_required}
                onChange={(event) =>
                  set("education_required", event.target.value as Form["education_required"])
                }
              >
                {EDUCATION_LEVELS.map((value) => (
                  <option key={value} value={value}>
                    {label(value)}
                  </option>
                ))}
              </select>
            )}
          </Field>
          <Field label="Minimum years of experience">
            {(props) => (
              <input
                {...props}
                type="number"
                min={0}
                max={50}
                step={1}
                value={form.min_years_experience}
                onChange={(event) => set("min_years_experience", event.target.value)}
              />
            )}
          </Field>
        </div>
        <Field label="Required skills" hint="Separate them with commas, as in: Python, SQL, dbt">
          {(props) => (
            <input
              {...props}
              value={form.required_skills}
              onChange={(event) => set("required_skills", event.target.value)}
            />
          )}
        </Field>
        <Field label="Preferred skills" hint="Separate them with commas">
          {(props) => (
            <input
              {...props}
              value={form.preferred_skills}
              onChange={(event) => set("preferred_skills", event.target.value)}
            />
          )}
        </Field>
        <div className="form-row">
          <Field label="Salary from">
            {(props) => (
              <input
                {...props}
                type="number"
                min={0}
                step="0.01"
                value={form.salary_min}
                onChange={(event) => set("salary_min", event.target.value)}
              />
            )}
          </Field>
          <Field label="Salary to">
            {(props) => (
              <input
                {...props}
                type="number"
                min={0}
                step="0.01"
                value={form.salary_max}
                onChange={(event) => set("salary_max", event.target.value)}
              />
            )}
          </Field>
          <Field label="Status" hint="An open job takes applications; a draft does not yet.">
            {(props) => (
              <select
                {...props}
                value={form.status}
                onChange={(event) => set("status", event.target.value as Form["status"])}
              >
                {NEW_JOB_STATUSES.map((value) => (
                  <option key={value} value={value}>
                    {label(value)}
                  </option>
                ))}
              </select>
            )}
          </Field>
        </div>
        <Field label="Description" hint="At least 20 characters.">
          {(props) => (
            <textarea
              {...props}
              required
              minLength={20}
              maxLength={20000}
              rows={8}
              value={form.description}
              onChange={(event) => set("description", event.target.value)}
            />
          )}
        </Field>
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
