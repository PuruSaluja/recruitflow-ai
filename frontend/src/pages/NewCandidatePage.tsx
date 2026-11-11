import { type ChangeEvent, type FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router";

import type { NewCandidate } from "../api/client";
import { useApi } from "../api/context";
import { Field, listFrom, optional, useSubmit } from "../components/Form";
import { ErrorMessage } from "../components/Status";
import { label } from "../format";
import { EDUCATION_LEVELS, SENIORITY_LEVELS } from "../options";

/** The resume formats the API reads. */
export const RESUME_TYPES = ".pdf,.txt,.md,application/pdf,text/plain,text/markdown";

type Form = {
  full_name: string;
  email: string;
  phone: string;
  location: string;
  current_title: string;
  years_experience: string;
  seniority: (typeof SENIORITY_LEVELS)[number];
  education_level: (typeof EDUCATION_LEVELS)[number];
  resume_text: string;
  skills: string;
};

const BLANK: Form = {
  full_name: "",
  email: "",
  phone: "",
  location: "",
  current_title: "",
  years_experience: "0",
  seniority: "mid",
  education_level: "bachelor",
  resume_text: "",
  skills: "",
};

/**
 * The request body. Skills left empty are left out, so the API takes them
 * from the resume, as it does for any candidate stored without skills.
 */
export function candidateFrom(form: Form): NewCandidate {
  const skills = listFrom(form.skills);
  return {
    full_name: form.full_name,
    email: form.email.trim(),
    phone: optional(form.phone),
    location: optional(form.location),
    current_title: optional(form.current_title),
    years_experience: Number(form.years_experience || 0),
    seniority: form.seniority,
    education_level: form.education_level,
    resume_text: form.resume_text,
    skills: skills.length > 0 ? skills : undefined,
  };
}

export function NewCandidatePage() {
  const client = useApi();
  const navigate = useNavigate();
  const [form, setForm] = useState<Form>(BLANK);
  const save = useSubmit();
  const read = useSubmit();
  const [found, setFound] = useState<{ file: string; skills: string[] } | null>(null);

  function set<K extends keyof Form>(key: K, value: Form[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function readResume(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setFound(null);
    const preview = await read.run(() => client.previewResume(file));
    if (!preview) return;
    // The text replaces what is in the box; skills fill it only when empty,
    // so skills already typed are kept, as the API keeps entered skills.
    setForm((current) => ({
      ...current,
      resume_text: preview.text,
      skills: current.skills.trim() ? current.skills : preview.skills.join(", "),
    }));
    setFound({ file: file.name, skills: preview.skills });
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const candidate = await save.run(() => client.createCandidate(candidateFrom(form)));
    if (candidate) navigate(`/candidates/${candidate.id}`);
  }

  return (
    <section>
      <h1>Add a candidate</h1>
      <form className="form" onSubmit={submit}>
        <Field
          label="Resume file"
          hint="A PDF, plain text or Markdown file up to 2 MB. Its text fills the resume below, and the skills it names fill an empty skills box."
        >
          {(props) => (
            <input {...props} type="file" accept={RESUME_TYPES} onChange={readResume} disabled={read.pending} />
          )}
        </Field>
        {read.pending && (
          <p className="muted" role="status">
            Reading the resume…
          </p>
        )}
        {read.error && <ErrorMessage error={read.error} />}
        {found && (
          <p className="muted" role="status">
            Read {found.file}.{" "}
            {found.skills.length > 0
              ? `It names ${found.skills.join(", ")}.`
              : "It names none of the skills the matcher knows; enter them below."}
          </p>
        )}
        <div className="form-row">
          <Field label="Full name">
            {(props) => (
              <input
                {...props}
                required
                minLength={2}
                maxLength={200}
                autoComplete="off"
                value={form.full_name}
                onChange={(event) => set("full_name", event.target.value)}
              />
            )}
          </Field>
          <Field label="Email">
            {(props) => (
              <input
                {...props}
                type="email"
                required
                maxLength={255}
                autoComplete="off"
                value={form.email}
                onChange={(event) => set("email", event.target.value)}
              />
            )}
          </Field>
        </div>
        <div className="form-row">
          <Field label="Phone">
            {(props) => (
              <input
                {...props}
                type="tel"
                maxLength={40}
                autoComplete="off"
                value={form.phone}
                onChange={(event) => set("phone", event.target.value)}
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
          <Field label="Current title">
            {(props) => (
              <input
                {...props}
                maxLength={200}
                value={form.current_title}
                onChange={(event) => set("current_title", event.target.value)}
              />
            )}
          </Field>
        </div>
        <div className="form-row">
          <Field label="Years of experience">
            {(props) => (
              <input
                {...props}
                type="number"
                min={0}
                max={60}
                step={1}
                value={form.years_experience}
                onChange={(event) => set("years_experience", event.target.value)}
              />
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
          <Field label="Education">
            {(props) => (
              <select
                {...props}
                value={form.education_level}
                onChange={(event) => set("education_level", event.target.value as Form["education_level"])}
              >
                {EDUCATION_LEVELS.map((value) => (
                  <option key={value} value={value}>
                    {label(value)}
                  </option>
                ))}
              </select>
            )}
          </Field>
        </div>
        <Field
          label="Skills"
          hint="Separate them with commas. Left empty, they are taken from the resume when it is saved."
        >
          {(props) => (
            <input {...props} value={form.skills} onChange={(event) => set("skills", event.target.value)} />
          )}
        </Field>
        <Field label="Resume" hint="At least 20 characters. Filled from the resume file, or paste it here.">
          {(props) => (
            <textarea
              {...props}
              required
              minLength={20}
              maxLength={100000}
              rows={12}
              value={form.resume_text}
              onChange={(event) => set("resume_text", event.target.value)}
            />
          )}
        </Field>
        {save.error && <ErrorMessage error={save.error} />}
        <div className="form-actions">
          <button type="submit" className="primary" disabled={save.pending || read.pending}>
            {save.pending ? "Saving…" : "Add candidate"}
          </button>
          <Link to="/jobs">Cancel</Link>
        </div>
      </form>
    </section>
  );
}
