import { type ChangeEvent, type FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router";

import { useApi } from "../api/context";
import {
  BLANK_CANDIDATE,
  CandidateFields,
  type CandidateForm,
  candidateFrom,
} from "../components/CandidateFields";
import { Field, useSubmit } from "../components/Form";
import { ErrorMessage } from "../components/Status";

/** The resume formats the API reads. */
export const RESUME_TYPES = ".pdf,.txt,.md,application/pdf,text/plain,text/markdown";

export function NewCandidatePage() {
  const client = useApi();
  const navigate = useNavigate();
  const [form, setForm] = useState<CandidateForm>(BLANK_CANDIDATE);
  const save = useSubmit();
  const read = useSubmit();
  const [found, setFound] = useState<{ file: string; skills: string[] } | null>(null);

  function set<K extends keyof CandidateForm>(key: K, value: CandidateForm[K]) {
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
        <CandidateFields
          form={form}
          set={set}
          resumeHint="At least 20 characters. Filled from the resume file, or paste it here."
        />
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
