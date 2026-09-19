# RecruitFlow AI

A full-stack recruiting platform: FastAPI backend, PostgreSQL data layer, React
frontend, and a scikit-learn matching layer that scores applicants against job
descriptions.

> **Status:** in development. The data layer and the job, candidate and
> application APIs are in place, including resume file ingestion and the
> pipeline stage-history trail, as are the skills taxonomy, the synthetic
> corpus the matcher will train on and the features it will read; the matching
> model itself, PDF reporting, and the frontend are landing incrementally.

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
end states are final for ordinary moves.

Stages change through `PATCH /api/v1/applications/{id}`. A move that skips a
stage, goes backwards, or leaves a finished application is refused with **409**
and a message saying which moves *are* possible; re-sending the current stage
is a no-op, so retries are safe.

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
accompany the `POST` that opens an application, a status-changing `PATCH` or a
reopen, to record who acted; there is no user system yet, so it is free text
rather than a foreign key, and it is only ever attached to a stage event, never
stored on the application itself.
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

## Synthetic corpus

There is **no real hiring data behind this project**. Every job, candidate and
outcome the matching layer trains on is generated by
[`data/generate_corpus.py`](data/generate_corpus.py), and nothing measured on it
is a finding about real people or real hiring.

```bash
python data/generate_corpus.py --out data/corpus
```

That writes `jobs.jsonl`, `candidates.jsonl`, `pairs.jsonl` and a
`manifest.json` holding the config used, the measured summary and the
interpreter that produced it. The files are not committed — like the model
artifacts, they are regenerated from the command above. A seed reproduces a
corpus exactly on one interpreter, but `random`'s sampling methods are not
guaranteed to draw the same across Python versions, which is why the manifest
records the version rather than assuming it.

### What the generator controls

- A candidate's **true profile** — role family, skills, experience, seniority,
  education — is drawn at random. What the profile *shows* is noisier: the
  skills list leaves some true skills out, and some are written under an alias,
  so the matcher has to cope with `Postgres` meaning `PostgreSQL`.
- **`gender` comes from its own random stream**, independent of every
  qualification. `age_band` is not independent — it follows education and years
  of experience, so the over-40 group really is the more experienced one.
- A pair's **`fit` is ground truth** computed from the true profile alone —
  skill coverage, a shortfall in years, distance in seniority, missing
  education — plus a random term for what no profile shows. It never sees
  demographics. `relevance` grades it 2/1/0 for ranking metrics.
- **`advanced` is a simulated past recruiter decision** — the kind of label a
  real system would learn from. It follows `fit`, but carries a planted bias.

Two biases are planted deliberately, so the fairness audit has something known
to find rather than noise to interpret:

| Plant | What it does | Removed by |
| --- | --- | --- |
| `historical_bias` | Women were advanced at `1 - historical_bias` times the rate of equally fit candidates | `--historical-bias 0` |
| `proxy_share` | Some women's resumes name a women-in-tech community group — a phrase a text model can learn as a stand-in for gender | `--proxy-share 0` |

Every resume carries a community line at the same rate whatever the candidate's
gender; only the wording differs.

### Measured on the default corpus

Real numbers from one run of the committed code — 200 jobs, 4,000 candidates,
10,000 labelled pairs on **CPython 3.14.6**, seed 42. Generation takes roughly
0.4 s (median of five runs on one laptop, spread 0.38–0.43 s), which is why the
corpus is regenerated rather than committed:

| | |
| --- | --- |
| Relevance 2 / 1 / 0 | 635 / 1,064 / 8,301 |
| Jobs with no relevant candidate on their slate | 0 (median 9 per job) |
| Jobs with nobody strong | 10 |
| Pool | 2,226 men, 1,563 women, 211 nonbinary; 3,595 under 40, 405 over |
| Advance rate overall | 15.1% |
| Advance rate, strong fits vs the rest | 69.3% vs 11.4% |
| Advance rate, men vs women | 16.5% vs 12.4% |
| **Women-to-men advance ratio** | **0.750** — the planted 0.25 bias, recovered |
| Women's resumes naming a women-in-tech group | 24.9% (0% of everyone else's) |

Re-running with `--historical-bias 0` moves the advance ratio to **1.000** and
lifts the overall rate to 16.7%; `--proxy-share 0` takes the proxy phrase to
**0%**. Both are the same corpus of people either way — the flags change only
the simulated decisions and the resume wording, never a qualification.

The ratio on any single corpus is a noisy estimate: across twelve seeds it runs
0.697–0.832 with the bias and 0.918–1.078 without it — the two ranges nearly
touch. Comparisons in the tests are therefore made against the same seed with
the bias removed, not against a fixed constant.

> **What this can and cannot establish.** A fairness audit on generated data can
> show that the audit itself works — that it recovers a bias of known size and
> reports none when none was planted. It cannot say anything about bias in real
> hiring, and no number here should be read as if it could.

## Matching features

The matcher reads each job-candidate pair as one row of numbers, built by
`PairFeaturiser` in [`backend/app/ml/matrix.py`](backend/app/ml/matrix.py). A
row has three blocks, always in this order:

1. **Eight engineered features** from fields a recruiter can see
   ([`app/ml/features.py`](backend/app/ml/features.py)):

   | Feature | Measures | Range |
   | --- | --- | --- |
   | `required_skill_coverage` | Share of the required skills the candidate lists | 0 to 1 |
   | `preferred_skill_coverage` | Share of the preferred skills the candidate lists | 0 to 1 |
   | `skill_jaccard` | Overlap between all the job's skills and the candidate's | 0 to 1 |
   | `experience_gap` | Years short of the job's minimum, full at five | 0 to 1 |
   | `experience_surplus` | Years past the minimum, full at five | 0 to 1 |
   | `seniority_distance` | Levels between the two, either way | 0 to 1 |
   | `seniority_direction` | Under-levelled (negative) or over-levelled (positive) | -1 to 1 |
   | `education_fit` | 1 when the requirement is met, less for each level short | 0 to 1 |

2. **`text_similarity`**, the cosine between the job's and the candidate's
   TF-IDF vectors ([`app/ml/text.py`](backend/app/ml/text.py)). A job's document
   is its title, description and skill lists; a candidate's is their current
   title, resume and listed skills.
3. **The candidate's own TF-IDF vector**, one column per term, named
   `resume:<term>`.

Skills are compared through the taxonomy, so `postgres` counts as
`PostgreSQL`, and a skill the taxonomy does not know still matches itself
whatever its case and spacing. A job with no required skills gives every
candidate full coverage rather than none, and two empty skill lists score no
overlap rather than a perfect one. Nothing in the matrix reads gender, age band
or a synthetic candidate's hidden true skills: tests change each of them and
require every row to stay identical.

Choices in the text half, each pinned by a test:

- **One spelling per word.** Every document is NFKC-normalised, cleared of
  zero-width characters and case-folded before it is split into terms, so
  `certiﬁed` with a ligature, `Ｐｙｔｈｏｎ` in full width and `café` in either
  Unicode form each give one term. Uploaded resumes are normalised on the way
  in, but resumes sent as JSON and job descriptions are not.
- **The tokeniser keeps skill names whole.** scikit-learn's default pattern
  drops `C++`, `C#` and `R` entirely and splits `Node.js`; this one keeps them,
  and still splits `Python/SQL` and hyphenated words. A `+` between two words
  does not split them, so `Python+SQL` stays one term.
- **No stop-word list.** scikit-learn's English list would delete `go`,
  `system`, `front`, `back` and `full`.
- **Repetition is damped.** A candidate's document names most skills twice,
  once in the resume and once in the skills list, so term counts are
  sublinear. A term must appear in at least two candidates' documents to be
  learned.
- **The vocabulary comes from the candidates alone**, each counted once however
  many slates they are on. Wording only jobs use ("We are hiring") can never
  match a resume; it would only lower each job's similarities by a different
  amount and add columns that are always zero.
- **The job's own terms are not a block.** They are the same for every
  candidate on a slate, so they cannot reorder one.

### The planted proxy is kept on purpose

The corpus plants a women-in-tech community line on some women's resumes, and
the text half treats it like any other wording: `resume:women` is a column. On
the default corpus it is non-zero in **1,017 of the 10,000 pairs**, and within
each relevance grade those pairs were advanced at 0.69, 0.87 and 0.74 times the
rate of the rest (grades 0, 1, 2); with `--historical-bias 0`, at 0.81, 1.07
and 0.96. One corpus is a noisy estimate — only 90 and 55 of those pairs sit
in grades 1 and 2 — so the same ratios were measured on seeds 1 to 12 as well:
medians of 0.83, 0.83 and 0.77 with the bias, and 1.02, 0.99 and 0.99 without
it. The term follows the planted recruiter bias, not candidate quality.

`women` is not the only such column: `bootcamp` and `member` also occur only in
the planted lines, in 126 and 113 of the 363 slate resumes that carry one
(`women` is in all 363). The proxy reaches a model through these columns, not
through `text_similarity`: swapping each planted line for the neutral line the
same person would otherwise have written moves that pair's similarity by
-0.0007 on average and never by more than 0.03, against a standard deviation
of 0.22 across all pairs.

The default configuration keeps these columns, and a model trained on past
decisions is free to learn them. That is deliberate. Removing words by hand
would be a mitigation nobody measured, and it relies on knowing which terms
stand in for gender: deleting `women` alone would leave the other two, and real
resumes carry stand-ins this corpus never generates, such as names and email
addresses, since its resumes contain no contact details. It would also hide
exactly what the fairness audit is built to find. Mitigations are tried in
that audit, with before-and-after numbers. Until then, expect a model trained
on `advanced` to score some women lower for this reason alone.

### Measured on the default corpus

Real numbers from one run of the committed code on the default corpus
(CPython 3.14.6, seed 42):

| | |
| --- | --- |
| Matrix for the 10,000 pairs | 10,000 × 355: 8 engineered, 1 similarity, 346 resume terms; 14% of cells non-zero |
| Vocabulary learned from | 3,680 distinct candidates on the slates |
| `text_similarity` by relevance grade 0 / 1 / 2 | 0.217 / 0.535 / 0.579 (mean) |
| Time to fit / to build all 10,000 rows | 0.12 s / 0.42 s (median of eleven runs, one laptop) |

Correlation of each of the first nine columns with the generator's
ground-truth `fit`, over the 10,000 pairs:

| Feature | Pearson r |
| --- | --- |
| `required_skill_coverage` | +0.704 |
| `preferred_skill_coverage` | +0.376 |
| `skill_jaccard` | +0.705 |
| `experience_gap` | -0.295 |
| `experience_surplus` | +0.086 |
| `seniority_distance` | -0.393 |
| `seniority_direction` | +0.077 |
| `education_fit` | +0.134 |
| `text_similarity` | +0.705 |

None of these is close to 1, by design. The features see the listed skills and
the written resume, while `fit` comes from the true profile plus a random term
for what no profile shows; closing that gap is the model's job.
`experience_surplus` and `seniority_direction` are weak because the generator
never rewards going past a job's minimum or level: it penalises only a
shortfall in years, and distance in seniority either way. They are kept for
real data, where being over-qualified may count for or against a candidate.

## License

MIT — see [LICENSE](LICENSE).
