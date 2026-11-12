import { type FormEvent, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";

import type { Candidate, CandidateChange } from "../api/client";
import { useApi, useLoad } from "../api/context";
import {
  candidateAsChange,
  CandidateFields,
  type CandidateForm,
  formFromCandidate,
} from "../components/CandidateFields";
import { changed, useSubmit } from "../components/Form";
import { ErrorMessage, Loaded } from "../components/Status";
import { parseId } from "../params";
import { NotFound } from "./NotFound";

function EditCandidateForm({ candidate }: { candidate: Candidate }) {
  const client = useApi();
  const navigate = useNavigate();
  const [form, setForm] = useState<CandidateForm>(() => formFromCandidate(candidate));
  const { pending, error, run } = useSubmit();
  const back = `/candidates/${candidate.id}`;

  function set<K extends keyof CandidateForm>(key: K, value: CandidateForm[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const after = candidateAsChange(form);
    const changes: CandidateChange = changed(candidateAsChange(formFromCandidate(candidate)), after);
    // The API fills an empty skill list from the resume only when the resume
    // comes with the change, so an empty skills box sends it along.
    if (after.skills.length === 0) changes.resume_text = after.resume_text;
    // With nothing changed there is nothing to ask of the API.
    if (Object.keys(changes).length === 0 || (await run(() => client.updateCandidate(candidate.id, changes)))) {
      navigate(back);
    }
  }

  return (
    <section>
      <p>
        <Link to={back}>← {candidate.full_name}</Link>
      </p>
      <h1>Edit candidate</h1>
      <form className="form" onSubmit={submit}>
        <CandidateFields form={form} set={set} resumeHint="At least 20 characters." />
        {error && <ErrorMessage error={error} />}
        <div className="form-actions">
          <button type="submit" className="primary" disabled={pending}>
            {pending ? "Saving…" : "Save changes"}
          </button>
          <Link to={back}>Cancel</Link>
        </div>
      </form>
    </section>
  );
}

export function EditCandidatePage() {
  const candidateId = parseId(useParams().candidateId);
  const [candidate, reload] = useLoad<Candidate | null>(`candidate:${candidateId}`, (api, signal) =>
    candidateId === null ? Promise.resolve(null) : api.getCandidate(candidateId, { signal }),
  );

  if (candidateId === null) return <NotFound what="candidate" />;
  if (candidate.state === "error" && candidate.error.status === 404) return <NotFound what="candidate" />;

  return (
    <Loaded value={candidate} what="the candidate" retry={reload}>
      {(data) => data && <EditCandidateForm candidate={data} />}
    </Loaded>
  );
}
