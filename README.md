# RecruitFlow AI

A full-stack recruiting platform: FastAPI backend, PostgreSQL data layer, React
frontend, and a scikit-learn matching layer that scores applicants against job
descriptions.

> **Status:** in development. The data layer and the job, candidate and
> application APIs are in place, including resume file ingestion and the
> pipeline stage-history trail; the matching model, PDF reporting, and the
> frontend are landing incrementally.

## Tech stack

| Layer    | Choice |
| -------- | ------ |
| API      | FastAPI, Pydantic v2, Uvicorn |
| Database | PostgreSQL 16, SQLAlchemy 2.0, Alembic |
| ML       | scikit-learn (TF-IDF + engineered features) |
| Reports  | ReportLab |
| Frontend | React 18, Vite, TypeScript |
| Tests    | pytest, httpx |

## Data model

Five tables, with native PostgreSQL enums and JSONB skill lists:

- **`jobs`** — requisitions: title, seniority, required/preferred skills, salary band, status.
- **`candidates`** — talent pool: resume text, skills, years of experience, education.
- **`applications`** — candidate → job pipeline state, unique per (job, candidate).
- **`application_stage_events`** — append-only audit trail of every pipeline
  move: from-status, to-status, when, and an optional free-text `changed_by`.
- **`match_scores`** — cached model output per (job, candidate, model version),
  including the per-feature breakdown used to explain a ranking.

## Local development

Requires Python 3.11+ and a running PostgreSQL 16 instance.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Unix
pip install -e ".[dev]"
cp .env.example .env          # then edit the connection strings
```

Create the schema:

```bash
alembic upgrade head
```

Run the API:

```bash
uvicorn app.main:app --reload --app-dir backend
```

Interactive docs are then at `http://localhost:8000/docs`.

## Tests

```bash
pytest
```

API tests run against a separate database, `RECRUITFLOW_TEST_DATABASE_URL`
(`recruitflow_test` in `.env.example`). Create it once; the suite applies the
Alembic migrations itself.

Every database test empties all tables first, so the suite refuses to start
unless the test database's name ends in `_test` and differs from
`RECRUITFLOW_DATABASE_URL`.

The resume normaliser's unit tests need no database:

```bash
pytest backend/tests/test_resume_service.py
```

`backend/tests/test_regressions.py` pins bugs found by probing the API with
awkward input — explicit nulls on required fields, JSONB storing a JSON `null`,
and `%`/`_` in search text acting as wildcards. Each pin fails if its fix is
reverted.

## API

| Method | Path | |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Liveness, including a real database round-trip |
| `GET` | `/api/v1/jobs` | List with pagination and filters |
| `POST` | `/api/v1/jobs` | Create a requisition |
| `GET` | `/api/v1/jobs/{id}` | Fetch one |
| `PATCH` | `/api/v1/jobs/{id}` | Partial update |
| `POST` | `/api/v1/jobs/{id}/close` | Close a requisition (idempotent) |
| `DELETE` | `/api/v1/jobs/{id}` | Delete, cascading to applications and scores |
| `GET` | `/api/v1/candidates` | List with pagination and filters |
| `POST` | `/api/v1/candidates` | Create a candidate |
| `GET` | `/api/v1/candidates/{id}` | Fetch one |
| `PATCH` | `/api/v1/candidates/{id}` | Partial update |
| `DELETE` | `/api/v1/candidates/{id}` | Delete, cascading to applications and scores |
| `GET` | `/api/v1/candidates/{id}/resume` | Stored resume text, as `text/plain` |
| `PUT` | `/api/v1/candidates/{id}/resume` | Upload a resume file, replacing the text |
| `POST` | `/api/v1/candidates/resume-preview` | Parse a resume file without storing it |
| `GET` | `/api/v1/applications` | List with pagination and filters |
| `POST` | `/api/v1/applications` | Apply a candidate to a job |
| `GET` | `/api/v1/applications/{id}` | Fetch one |
| `PATCH` | `/api/v1/applications/{id}` | Update notes, source or pipeline stage |
| `GET` | `/api/v1/applications/{id}/stage-events` | Full pipeline history, oldest first |
| `POST` | `/api/v1/applications/{id}/reopen` | Put a rejected/withdrawn application back to `applied` |
| `DELETE` | `/api/v1/applications/{id}` | Delete an application |

Candidate listings filter on `seniority`, `education_level`,
`min_years_experience`, `max_years_experience`, `location` (substring), `q`
(name, email or current title) and `skill`. `skill` is repeatable and
conjunctive — `?skill=python&skill=airflow` returns only candidates who have
both — and matching is case-insensitive, so stored `"Python"` answers
`?skill=python`.

Application listings filter on `job_id`, `candidate_id`, `status` and `source`.

## Hiring pipeline

An application moves `applied` → `screening` → `interview` → `offer` → `hired`.
`rejected` and `withdrawn` are reachable from any live stage, and all three
end states are final.

Stages change through `PATCH /api/v1/applications/{id}`. A move that skips a
stage, goes backwards, or leaves a finished application is refused with **409**
and a message naming the stages that *are* reachable; re-sending the current
stage is a no-op, so retries are safe.

A rejected or withdrawn application can be put back into play through
`POST /api/v1/applications/{id}/reopen`, which is deliberately a separate
action rather than a loosened `PATCH` rule — the 409 for a rejected/withdrawn
`PATCH` names it as the way back in. Reopening returns the same application to
`applied` instead of creating a new one, since a candidate holds only one
application per job; the stage event it writes carries the real `from_status`
(`rejected` or `withdrawn`), so that stint's duration stays answerable from the
data. A `hired` application cannot be reopened, and reopening still needs the
requisition to be **open** — both refuse with 409.

Two more 409s guard creation: only an **open** requisition accepts new
applications, and a candidate can apply to a given job only once. That second
rule is enforced by a unique constraint rather than a pre-check, so two
simultaneous requests cannot both succeed. Closing a job stops new applications
but leaves the ones already in flight editable.

Every move — including the opening one at creation — appends a row to
`application_stage_events`, retrievable through
`GET /api/v1/applications/{id}/stage-events`. A `changed_by` string can
accompany a `POST` or a status-changing `PATCH` to record who acted; there is
no user system yet, so it is free text rather than a foreign key, and it is
only ever attached to a stage event, never stored on the application itself.
A move that is refused or a no-op (resending the current status, a PATCH that
only touches `notes`/`source`) records nothing.

Two recruiters acting on the same application at the same moment used to end
with the later write winning silently and both moves recorded in the trail.
Applications now carry a version counter, so the update that lost the race
matches no row and comes back as a **409** asking the caller to re-read and
retry; its stage event is rolled back with it. The counter is internal — it is
not in the API responses and cannot be sent in a `PATCH`.

## Resume ingestion

`PUT /api/v1/candidates/{id}/resume` accepts a PDF or a plain-text/Markdown
file up to 2 MB and replaces the candidate's `resume_text` with the extracted
content. The format is decided by sniffing the file's magic bytes, not its
name, so a PDF mailed as `resume.txt` is still read as a PDF.

Extracted text is normalised before it is stored, because every artefact left
in it becomes a token for the matching layer downstream:

- Unicode is NFKC-normalised, folding ligatures (`ﬁ` → `fi`), full-width
  characters and non-breaking spaces.
- Zero-width characters, soft hyphens and C0/C1 control bytes are removed.
  PDF exporters emit control bytes wherever a glyph is missing from the
  font's encoding.
- Leading bullet glyphs — including U+F0B7, the private-use codepoint Word
  uses for Symbol-font bullets — become `- `. Only line-leading glyphs are
  rewritten, so a mid-sentence `·` stays punctuation.
- Words split across a line break (`respon-\nsibility`) are rejoined.
- Page footers (`Page 2 of 4`) are dropped. A bare number on its own line is
  deliberately kept: in a resume that is far more often a date than a page
  number.
- Runs of spaces, tabs and blank lines are collapsed.

Failures are 400s with a message written for the uploader — an encrypted PDF,
a `.docx` (zip-based formats are rejected with a pointer to export as PDF),
or a scan with no text layer each say so specifically.

## License

MIT — see [LICENSE](LICENSE).
