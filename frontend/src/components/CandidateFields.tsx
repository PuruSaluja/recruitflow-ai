import type { Candidate, CandidateChange, NewCandidate } from "../api/client";
import { label } from "../format";
import { EDUCATION_LEVELS, SENIORITY_LEVELS } from "../options";
import { Field, listFrom, optional } from "./Form";

/** A candidate as their form holds them: every box's text, and each select's value. */
export type CandidateForm = {
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

export const BLANK_CANDIDATE: CandidateForm = {
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
 * The body that creates a candidate. Skills left empty are left out, so the
 * API takes them from the resume, as it does for any candidate stored without
 * skills.
 */
export function candidateFrom(form: CandidateForm): NewCandidate {
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

/** The form showing a candidate as they are stored. */
export function formFromCandidate(candidate: Candidate): CandidateForm {
  return {
    full_name: candidate.full_name,
    email: candidate.email,
    phone: candidate.phone ?? "",
    location: candidate.location ?? "",
    current_title: candidate.current_title ?? "",
    years_experience: String(candidate.years_experience),
    seniority: candidate.seniority,
    education_level: candidate.education_level,
    resume_text: candidate.resume_text,
    skills: (candidate.skills ?? []).join(", "),
  };
}

/**
 * The candidate a form describes, as a change would set them: every field,
 * with an empty optional one as null, which clears it where leaving it out
 * would not.
 */
export function candidateAsChange(form: CandidateForm) {
  return {
    full_name: form.full_name,
    email: form.email.trim(),
    phone: optional(form.phone) ?? null,
    location: optional(form.location) ?? null,
    current_title: optional(form.current_title) ?? null,
    years_experience: Number(form.years_experience || 0),
    seniority: form.seniority,
    education_level: form.education_level,
    resume_text: form.resume_text,
    skills: listFrom(form.skills),
  } satisfies Required<CandidateChange>;
}

/** A candidate form's fields, the same for a new candidate and for one being changed. */
export function CandidateFields({
  form,
  set,
  resumeHint,
}: {
  form: CandidateForm;
  set: <K extends keyof CandidateForm>(key: K, value: CandidateForm[K]) => void;
  resumeHint: string;
}) {
  return (
    <>
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
              onChange={(event) => set("seniority", event.target.value as CandidateForm["seniority"])}
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
              onChange={(event) => set("education_level", event.target.value as CandidateForm["education_level"])}
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
      <Field label="Resume" hint={resumeHint}>
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
    </>
  );
}
