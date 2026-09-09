# RecruitFlow AI

A full-stack recruiting platform: FastAPI backend, PostgreSQL data layer, React
frontend, and a scikit-learn matching layer that scores applicants against job
descriptions.

> **Status:** in development. The data layer and job requisition API are in
> place; candidate/application endpoints, the matching model, PDF reporting,
> and the frontend are landing incrementally.

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

## License

MIT — see [LICENSE](LICENSE).
