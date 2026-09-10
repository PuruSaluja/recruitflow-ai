# RecruitFlow AI

A full-stack recruiting platform: FastAPI backend, PostgreSQL data layer, React
frontend, and a scikit-learn matching layer that scores applicants against job
descriptions.

> **Status:** in development. The data layer, job requisition API and candidate
> API (including resume file ingestion) are in place; application endpoints,
> the matching model, PDF reporting, and the frontend are landing
> incrementally.

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

Four tables, with native PostgreSQL enums and JSONB skill lists:

- **`jobs`** — requisitions: title, seniority, required/preferred skills, salary band, status.
- **`candidates`** — talent pool: resume text, skills, years of experience, education.
- **`applications`** — candidate → job pipeline state, unique per (job, candidate).
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

Candidate listings filter on `seniority`, `education_level`,
`min_years_experience`, `max_years_experience`, `location` (substring), `q`
(name, email or current title) and `skill`. `skill` is repeatable and
conjunctive — `?skill=python&skill=airflow` returns only candidates who have
both — and matching is case-insensitive, so stored `"Python"` answers
`?skill=python`.

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
