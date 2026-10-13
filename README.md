# RecruitFlow AI

A full-stack recruiting platform: FastAPI backend, PostgreSQL data layer, React
frontend, and a scikit-learn matching layer that scores applicants against job
descriptions.

> **Status:** in development. The data layer and the job, candidate and
> application APIs are in place, including resume file ingestion and the
> pipeline stage-history trail, as is the matching layer: the skills taxonomy,
> the synthetic corpus the matcher trains on, its features, the calibrated
> classifier, endpoints that score a job's candidate pool and cache the
> ranking, a held-out evaluation, sentence-transformer embeddings of every
> candidate and job, cached in PostgreSQL, a measured comparison of lexical,
> dense and hybrid retrieval, a measured cross-encoder re-rank of the
> hybrid's shortlist, and a fairness audit of the matcher's shortlists, whose
> findings the served matcher is now trained to counter, and PDF reports of a
> job's shortlist and a candidate's scorecard, and a web client that lists
> jobs, shows a job's ranked candidates and scores them, and shows a
> candidate with their applications.

## Tech stack

| Layer    | Choice |
| -------- | ------ |
| API      | FastAPI, Pydantic v2, Uvicorn |
| Database | PostgreSQL 16, SQLAlchemy 2.0, Alembic |
| ML       | scikit-learn (TF-IDF + engineered features); sentence-transformers (embeddings, optional) |
| Reports  | ReportLab |
| Frontend | React 19, React Router 8, Vite 8, TypeScript 5.9; API types generated from the OpenAPI schema |
| Tests    | pytest, httpx2; Vitest for the web client |

## Data model

Seven tables, with native PostgreSQL enums and JSONB skill lists:

- **`jobs`** — requisitions: title, seniority, required/preferred skills, salary band, status.
- **`candidates`** — talent pool: resume text, skills, years of experience, education.
- **`applications`** — candidate → job pipeline state, unique per (job, candidate).
- **`application_stage_events`** — append-only audit trail of every pipeline
  move: from-status, to-status, when, and an optional free-text `changed_by`.
- **`match_scores`** — cached model output per (job, candidate, model version),
  including the per-feature breakdown used to explain a ranking.
- **`candidate_embeddings`**, **`job_embeddings`** — one float32 vector per
  (owner, encoder), with the hash of the text it was made from; see
  [Embeddings](#embeddings).

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

Run the web client (Node 22.22 or later, which React Router 8 needs) beside
it:

```bash
cd frontend
npm ci
npm run dev
```

It is served at `http://localhost:5173`, and the dev server forwards `/api` to
the API on `127.0.0.1:8000` (`RECRUITFLOW_API_TARGET` points it elsewhere), so
the browser talks to one origin.

| Page | Address | Shows |
| ---- | ------- | ----- |
| Jobs | `/jobs` | Every requisition, newest first, 20 a page; title search and a status filter, kept in the address |
| Job | `/jobs/{id}` | The job's facts, skills and requirement notes, then its ranking under the current matcher, 25 a page: score, required skills held, experience, links to each scorecard and to the shortlist PDF. **Score all candidates** scores everyone on file and reloads the ranking |
| Candidate | `/candidates/{id}` | Contact details, experience, skills, resume text, and their applications, latest first, linked to each job |

An error from the API is shown with its own message (a missing matcher, a
database that is down, a scoring run that lost a race), with a button to
load again where a page or its ranking failed to load; an
id that does not exist, or that no record could have, is a not-found page.

The client's types in `frontend/src/api/schema.ts` are generated from
`frontend/openapi.json`, the API's schema. After changing the API, refresh
both:

```bash
python scripts/export_openapi.py
cd frontend && npm run gen:api
```

The suite fails while `openapi.json` lags the API in anything the types are
generated from (a reworded description is only a warning, so a new FastAPI or
pydantic release cannot fail it), and CI fails while `schema.ts` lags
`openapi.json`.

Without the dev server, build the client and the API serves it:

```bash
cd frontend && npm run build && cd ..
uvicorn app.main:app --app-dir backend
```

The app is then at `http://localhost:8000`. Built files are sent as they are,
hashed assets with a year-long cache; every other page address (`/jobs/12`)
gets `index.html`, so a reload or a shared link works. The API and its docs
are matched first, and an unknown API path or a missing file is still a JSON
404. `RECRUITFLOW_WEB_CLIENT_DIR` points at another build, or, empty, turns
serving it off; with no build in `frontend/dist` nothing is served at `/`.

The app's own log lines go to stderr as `time LEVEL logger: message`, from
`RECRUITFLOW_LOG_LEVEL` up (default `INFO`); uvicorn's lines keep their own
format.

## Tests

```bash
pytest
```

API tests run against a separate database, `RECRUITFLOW_TEST_DATABASE_URL`
(`recruitflow_test` in `.env.example`). Create it once; the suite applies the
Alembic migrations itself, and checks that they build the schema the models
describe.

Every database test empties all tables first, so the suite refuses to start
unless the test database's name ends in `_test` and differs from
`RECRUITFLOW_DATABASE_URL`.

The web client's tests (Vitest) need neither the API nor a database:

```bash
cd frontend
npm test
npm run build      # type-checks, then builds
```

The end-to-end tests drive the built client in a real browser against a
running API: build it, serve it with uvicorn as above (a migrated database
and a trained matcher), then

```bash
cd frontend
E2E_CHANNEL=msedge npm run e2e -- --project=chromium --project=mobile
```

`E2E_CHANNEL` uses an installed Edge or Chrome (`chrome`); without it, run
`npx playwright install` first for Playwright's own browsers. They create a
job, three candidates and an application through the API and delete them
afterwards, and check: finding the job, scoring its candidates, that both PDF
links return PDFs, opening a candidate, reloading a page address, the
not-found pages, no serious or critical accessibility findings from axe-core
in light or dark, and no sideways scroll, at desktop and phone sizes
(`E2E_BASE_URL` points them at another server). CI runs them in Chromium,
Firefox, WebKit and a phone-sized Chromium, and the client's own job on Node
22.22.0 and 24.

The resume normaliser's unit tests need no database:

```bash
pytest backend/tests/test_resume_service.py
```

`backend/tests/test_regressions.py` pins bugs found by probing the API with
awkward input — explicit nulls on required fields, JSONB storing a JSON `null`,
`%`/`_` in search text acting as wildcards, a stage change overwriting a
simultaneous one, non-positive ids, ids and offsets past what PostgreSQL can
hold, writes that lose a race to another request or deadlock with one, values
the database or the response cannot hold, oversized bodies read in full
before they were refused, and a listing's total counted in a separate
statement from its page. Each pin fails if its fix is reverted.

### Coverage

```bash
pytest --cov --cov-report=term-missing:skip-covered
```

Coverage counts lines and branches in `backend/app` and in `scripts`, the
command-line entry points that train, evaluate and audit the matcher and write
the sample reports. Scripts the tests run as a subprocess are counted only
where a test also calls them in-process. Measured on 2026-10-12 over the whole
suite (1,658 tests, every one passing, with the embeddings extra installed so
the real models load):

| Area | Lines | Branches | Covered |
| ---- | ----- | -------- | ------- |
| `app/ml` (features, training, ranking, retrieval, fairness) | 1,185 of 1,185 | 266 of 266 | 100% |
| `app/services` (scoring, reports, resumes, embeddings, requirements) | 438 of 439 | 103 of 104 | 99.6% |
| `app/api` (routes, body limit, errors) | 582 of 582 | 125 of 126 | 99.9% |
| `app/schemas` | 280 of 282 | 15 of 18 | 98.3% |
| `app/db`, `app/models` and top level | 357 of 357 | 14 of 14 | 100% |
| `scripts` | 1,110 of 1,140 | 224 of 232 | 97.2% |
| **Total** | **3,952 of 3,985** | **747 of 760** | **99.0%** |

Of the 33 lines missed, 7 are the calls to `main()` under the scripts'
`if __name__ == "__main__"`, and 18 are six scripts' handling of an output
that turns unwritable during a run: each checks its output path before it
starts (`app/cli.py`), and the tests cover that check. The rest are conditions
the tests do not stage (a server that has pgvector, a machine without torch),
the timing script's loaders of the real models, which its tests replace with
fakes, and three lines in the schemas and the resume decoder. CI's main job
prints the same report on every push; it leaves the embeddings extra out, so
the 21 tests that load the real models skip there and it misses more lines.

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
| `POST` | `/api/v1/jobs/{id}/score` | Score a candidate pool against a job and cache the ranking |
| `GET` | `/api/v1/jobs/{id}/matches` | The cached ranking under the current matcher, paginated |
| `GET` | `/api/v1/jobs/{id}/report.pdf` | The cached ranking's shortlist as a PDF |
| `GET` | `/api/v1/jobs/{id}/candidates/{cid}/scorecard.pdf` | One candidate's scorecard for the job as a PDF |

Candidate listings filter on `seniority`, `education_level`,
`min_years_experience`, `max_years_experience`, `location` (substring), `q`
(name, email or current title) and `skill`. `skill` is repeatable and
conjunctive — `?skill=python&skill=airflow` returns only candidates who have
both — and matching is case-insensitive, so stored `"Python"` answers
`?skill=python`.

Application listings filter on `job_id`, `candidate_id`, `status` and `source`.

Every timestamp is ISO 8601 in UTC, ending in `Z`, such as
`2026-09-25T01:48:29.260762Z`.

### Limits and errors

- Every id, in a path, a body or a query, is at most 2,147,483,647, the
  largest PostgreSQL `INTEGER`, and `offset` at most
  9,223,372,036,854,775,807, the largest `BIGINT`.
- A candidate's `resume_text` holds at most 100,000 characters, and their
  `skills` at most 100 entries of at most 100 characters each, counted after
  spacing is tidied.
- A job's `description` holds at most 20,000 characters, and its
  `required_skills` and `preferred_skills` at most 100 entries each, of at
  most 100 characters as sent.
- Salaries fit their `NUMERIC(10, 2)` columns: at most 99,999,999.99, in
  whole cents.
- Text may not contain the NUL character (U+0000), which no PostgreSQL text
  column can store.

Breaking any of these in a path, body or query is a **422**; an uploaded
resume whose text is too long is refused like any other unusable file, with a
400. A 422 names each problem's type, location and message but does not repeat
the rejected value, so a resume refused for its length gets a short answer. A
value only the database can refuse, such as the NUL character, is a 422 with
`code: "invalid_value"`.

A request body holds at most 2,162,688 bytes: the 2 MB resume upload limit
plus 64 KiB for the multipart framing around the file. A larger one is a
**413** with `code: "request_too_large"`, decided before the app sees any of
the body when the request declares its length, and as soon as the limit is
passed when it does not; the connection is then closed. Before the 413 goes
out, the rest of the body is read and thrown away, up to 16 MiB in all: a
server that closes a connection while the client is still sending resets it,
and the client never sees the answer. Past 16 MiB the rest is left unread, so
a much larger upload can still end in a reset. A client sending
`Expect: 100-continue`, as curl does for large uploads, is answered at once
and sends none of the file.

A write that loses a race to another request says what happened: **404** if
the other request deleted the row, whether just before this one committed or
just after, before it could be read back, and **409** if it changed the row.
Where the answer depends on the row as the write finds it, the write holds the
row from reading it to committing, so a second request writing it waits and
then answers as it would have one after the other: a job `PATCH` whose salary
band someone else just changed into one this edit would invert is a **422**,
and a `DELETE` of a job or candidate beaten by another `DELETE` is a **404**.
An application whose job or candidate is deleted while it is being created is
a **404** too, the foreign key refusing it. Creating one holds neither row
while it checks them, so it does not wait on an edit of the job, and a repeat
application no longer deadlocks with a delete of either. None of these is
reported as the database being unavailable, which is kept for real outages
(**503**).

Deleting a job or candidate takes its applications, their stage events and
its cached scores with it. If one of those applications is edited while the
delete is in flight, the delete is a **409**: it would remove a change the
caller has not seen. If another request deletes one instead, a delete of the
other side of it say, that is no conflict, and the delete tries again, up to
three times in all. Until 2026-09-29 that was a 409 too. The same 120 s mixed
soak from 12 threads, run against the old code and then the new on one
throwaway database, gave 14 such 409s in 4,405 requests before and none in
4,592 after, and the server's 110 warnings about deleting rows another delete
had already removed went to none. Stage events now go by their foreign key's
`ON DELETE CASCADE`. Scores do not: letting the cascade remove them locks the
job or candidate first, and a replayed delete then deadlocked with a writer
holding a score. They go first instead, in one statement, under the job's or
candidate's lock. So a delete of the other side of a pair waits for the pair's
score rather than finding it gone, and a job's delete no longer loads every
score to delete it by id: with 11,964 scores, six deletes each on the old and
the new code, alternating on one throwaway database, a job delete took a
median 719 ms before (700–746) and 62 ms after (54–80), the job staying
locked for most of that time.

A 503 with `code: "db_unavailable"` means the database could not be reached or
could not serve the request: a refused or dropped connection, a deadlock, a
cancelled statement, or no pooled connection free. Any other database error is
a fault in the server's own code and is a **500** with
`code: "internal_error"`.

So is any other error no handler claims. It is answered from inside the CORS
middleware, so a browser client can read the 500 rather than seeing a network
error it cannot tell from the server being down, and the error is logged with
its traceback. An error raised once a streamed answer has started cannot be
replaced by a 500; that answer is cut short.

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

The extracted text, once normalised as below, may be at most 100,000
characters, the same cap `resume_text` has in a create or update. The 2 MB
limit cannot stand in for it, because it bounds the file's bytes: a
20,414-byte PDF generated for the tests extracts to 108,028 characters.

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

### The planted proxy

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

`PairFeaturiser` keeps these columns, so the fairness audit can measure what
a model trained on past decisions does with them: it scores every pair
carrying the planted line lower, and deleting the three words closes only a
small part of the resulting gap. Removing words by hand relies on knowing
which terms stand in for gender, and real resumes carry stand-ins this corpus
never generates, such as names and email addresses. So the matcher the app
serves does not delete words. It never reads a line headed `Community:` at
all, and it is trained on reweighed pairs (see *Training pipeline* and
*Fairness audit* below).

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

## Training pipeline

`backend/app/ml/train.py` turns a corpus into a calibrated matcher:
`training_data` resolves a corpus's labelled pairs into objects and parallel
label arrays, `split_by_job` holds out a share of jobs with their whole
slates, and `make_matcher`/`train_matcher` build and fit a pipeline of
featuriser → `LogisticRegression`, wrapped in `CalibratedClassifierCV`. The
featuriser is a `PairFeaturiser` that skips every line headed `Community:`
(`EXCLUDED_HEADINGS`), and `train_matcher` reweighs the training pairs.

**The training target is `advanced`**, the past recruiter's call, not
`relevance`. `relevance` is ground truth derived from `true_skills`, which no
real recruiting system has, and the planted bias lives entirely in
`advanced` — a model fitted on ground truth would leave the fairness audit
nothing to find. `relevance` is carried beside the pairs for the ranking
evaluation and never fitted.

**The bias in that record is counteracted, not inherited.** Trained plainly
on `advanced`, the matcher shortlisted qualified women 5.0 points less often
than qualified men, averaged over nine audit runs, mostly through the planted
community line. So the served matcher never reads a community line, and
`train_matcher` weights each training pair so that gender and the past
decision are independent (Kamiran and Calders' reweighing). Gender is read
only to compute those weights, from the training corpus. It is never a
feature, and scoring a candidate never needs it. The *Fairness audit* below
gives the result.

**The split is by job**, so a slate is never torn in half — ranking metrics
are per job over a whole slate. The cost is candidate overlap: 84.5% of the
default corpus's 1,844 held-out candidates also sit on a training slate.
Measured on the served matcher, that overlap does not inflate the held-out
score: on the 392 held-out pairs whose candidate is on no training slate, a
matcher trained without any held-out candidate scores 0.8431 ROC-AUC against
this one's 0.8452, while leaving those candidates out of training discards
45.7% of the training pairs.

**Calibration wraps the whole pipeline** rather than sitting inside it, so
each fold learns its own vocabulary instead of calibrating against resumes it
was fitted on. **Sigmoid, not isotonic**: they calibrate this corpus about
equally well (Brier 0.1072 against 0.1068, log loss 0.3538 against 0.3658),
but sigmoid is strictly monotone — isotonic returned 851 distinct
probabilities over 2,500 held-out pairs against sigmoid's 2,500, and every
tie is two candidates a ranking then has to separate arbitrarily. Nothing is tuned: the
estimators keep scikit-learn's defaults.

### Persisting the matcher

Model binaries do not belong in git, so `backend/app/ml/artifacts/` is
gitignored in full. `scripts/train_matcher.py` trains a matcher on a
synthetic corpus (generated in memory — it does not read the JSON Lines
files `data/generate_corpus.py` writes) and persists it with
[`app/ml/artifact.py`](backend/app/ml/artifact.py):

```
python scripts/train_matcher.py
```

This writes `backend/app/ml/artifacts/matcher.joblib` (overridable with
`--out`, and via `RECRUITFLOW_MODEL_PATH`/`Settings.model_path`) plus a
`matcher.manifest.json` beside it, recording the corpus config, the split,
the training label, measured held-out metrics, and the interpreter —
`random`'s draws and the calibration folds both depend on it, the way the
corpus manifest already records that for the corpus itself. Anything that
needs a trained matcher loads it through `app.ml.artifact`, which raises
`MatcherUnavailable` naming this command when the file is missing or cannot
be unpickled, rather than letting either surface as a stack trace; the
matching endpoints turn that into a 503.

The CLI refuses to write a matcher rather than silently shipping a bad one:
a training split with only one label present fails scikit-learn's solver
with an opaque message ("needs samples of at least 2 classes"), so it is
checked for up front and named in plain language; a split where the rarer
label has fewer examples than `CalibratedClassifierCV`'s five folds would
otherwise only draw a scikit-learn warning and calibrate on folds too thin to
trust, so that is refused too; and an out-of-range or non-numeric
`--test-share` (`nan` included) is validated before it reaches `split_by_job`,
rather than surfacing as `round()`'s "cannot convert float NaN to integer".
All three are exercised in `backend/tests/test_train_matcher.py` against
corpus shapes measured to trigger them.

### Measured on the default corpus

Real numbers from one run of the committed code (CPython 3.14.6, default
corpus, `--test-share 0.25 --split-seed 0`, both scikit-learn's defaults):

| | |
| --- | --- |
| Trained on / held out | 7,500 pairs across 150 jobs / 2,500 pairs across 50 jobs |
| Held-out ROC-AUC vs `advanced` | 0.7832 |
| Held-out ROC-AUC vs `relevance >= 1` (reference only; never fitted) | 0.9435 |
| Mean predicted probability vs training base rate | 0.1503 vs 0.1508 |

`relevance`'s ROC-AUC is higher because it is the easier, unbiased target the
generator itself defines; the matcher is not trained on it. These numbers are
for reference; the ranking task they stand in for is measured below, and so
is what the planted bias in `advanced` does to the matcher's shortlists
(*Fairness audit*).

### Ranking quality on held-out jobs

`scripts/evaluate_matcher.py` scores the persisted matcher's own held-out
jobs (after checking it recomputes the ROC-AUC its manifest recorded) and
grades each slate against `relevance`, which the matcher never saw. From one
run on the default matcher, written to
[`docs/model_evaluation.json`](docs/model_evaluation.json):

| 50 held-out jobs, 2,500 pairs | NDCG@10 | P@5 |
| --- | --- | --- |
| Matcher | 0.8110 (95% CI 0.7754–0.8444) | 0.7960 (0.7400–0.8480) |
| Random ranking (exact expectation) | 0.1638 | 0.1744 |
| `required_skill_coverage` alone | 0.6533 | 0.6419 |
| Same pipeline fitted on `relevance >= 1` (reference; no real system has it) | 0.8441 | 0.8400 |

Over ten splits of the same corpus (the matcher's own and nine retrained),
NDCG@10 is 0.8102 ± 0.0198 and P@5 0.8084 ± 0.0199 (mean ± std). The metric
definitions, the tie handling and what these numbers cannot show are in
[`docs/model_evaluation.md`](docs/model_evaluation.md).

### Lexical, dense and hybrid retrieval

The same script measures three scorers fitted to no label
(`app/ml/retrieval.py`): BM25 over the candidates' documents with the job's
as the query, cosine between the documents' embeddings from the pretrained
model, and the two fused by reciprocal rank fusion (`k = 60`) within each
slate. Every setting is the textbook default; none was tuned. Since none has
seen this corpus's labels, they are also measured on all 200 jobs:

| NDCG@10 | 50 held-out jobs | All 200 jobs |
| --- | --- | --- |
| BM25 | 0.5081 | 0.5333 |
| Dense | 0.6637 | 0.6566 |
| Hybrid (RRF) | 0.6259 | 0.6392 |

**Fusion does not beat dense retrieval here.** Job by job, the hybrid beats
BM25 by +0.1059 NDCG@10 over all jobs (95% CI +0.0886 to +0.1231) but trails
dense by 0.0174 (−0.0344 to +0.0002), and on the held-out jobs by 0.0378
(−0.0655 to −0.0096). The two sides rank each slate much alike (Spearman
0.8439 on average), BM25 is much the weaker, and equal-weight fusion gives it
as much say as dense: fusion beat dense on 46 of the 53 jobs where BM25 ranked
at least as well, but on only 41 of the other 147. All three sit well below
the trained matcher. Details and P@5 are in
[`docs/model_evaluation.md`](docs/model_evaluation.md).

### Cross-encoder re-rank

A two-stage ranking: the hybrid keeps the top 20 of each slate (chosen on
the training jobs, by recall, before any re-rank was run), and a pretrained
cross-encoder, ms-marco-MiniLM-L6-v2 pinned to one revision
(`app/ml/cross_encoder.py`), re-orders those by reading each resume beside
the job description. A resume is scored by its best 128-word-piece passage,
since unrelated text around a match drowns it: a one-line Python resume
after 60 lines of warehouse work scored -10.86 read as one window (the
warehouse lines alone, -11.16) and 0.14 read as passages (alone, -11.36).

**On this corpus it makes the ranking worse and slower.** NDCG@10 falls from
the hybrid's 0.6392 to 0.5611 over all 200 jobs (paired difference −0.0781,
95% CI −0.1021 to −0.0527, worse on 138 jobs and better on 61), and from
0.6259 to 0.5788 on the held-out jobs (−0.0472, −0.0959 to +0.0006). It
still beats BM25 alone. On a laptop CPU the cross-encoder takes 1.6 to 1.8 s
per slate, about 95% of the request, against under 0.1 s for the first
stage (`scripts/time_rerank.py`, `docs/rerank_latency.json`). The model was
trained on web-search queries and passages, not job descriptions and
resumes; whether that is the cause was not tested, and nothing was tuned
after seeing these numbers. **The API does not use the re-rank**, and on
these results it should not.

On this synthetic corpus the first stage's top 20 is very nearly a filter
for the job's own role family: each slate has 20 candidates from it, BM25's
top 20 is exactly those on every slate, the hybrid's is 99.3% from the
family over all 200 jobs, and 96% of the relevant candidates are in the
family. So the re-rank orders candidates almost entirely within one family,
and the corpus barely tests finding a good candidate from another
(`first_stage.same_family` in the JSON).

### Fairness audit

`scripts/audit_fairness.py` measures who reaches the top 5 and top 10 of each
slate, by gender (against men) and by age band (against under 40). For each
group it reports the selection rate, the disparate impact ratio with the
four-fifths rule, and the equal-opportunity difference among qualified
candidates, each with a bootstrap interval over jobs. Every job is scored by a
matcher trained without it (four job folds), so all 200 jobs count. That
interval covers only which jobs were sampled, so the whole audit is repeated
over other fold seeds and other corpora: nine runs that the served matcher's
configuration was chosen on
([`docs/fairness_audit.json`](docs/fairness_audit.json)), and eight fresh runs
that played no part in the choice
([`docs/fairness_audit_fresh.json`](docs/fairness_audit_fresh.json)).

Women against men, equal-opportunity difference (negative: qualified women
shortlisted less often than qualified men), mean over the runs and how many
runs' intervals exclude zero:

| | Top 5, 9 runs | Top 10, 9 runs | Top 5, 8 fresh runs | Top 10, 8 fresh runs |
| --- | --- | --- | --- | --- |
| Past decisions the matcher learns from (every pair) | −0.140 (9) | | −0.151 (8) | |
| Matcher trained plainly on them | −0.050 (5) | −0.046 (6) | −0.046 (6) | −0.040 (6) |
| **Served matcher**: community line unread, pairs reweighed | **−0.011 (0)** | **−0.018 (0)** | **−0.013 (0)** | **−0.010 (0)** |

The past decisions are biased against women in every run, as planted, and a
matcher trained plainly on them carries part of that into its shortlists,
mostly through the planted community line. The served matcher never reads
that line and is trained on pairs reweighed so that gender and the past
decision are independent. It shows no gap for women that can be told from
zero in any of the 17 runs, nor for nonbinary candidates in the fresh ones.
Its NDCG@10 is never measurably lower than the plain matcher's. Each fix
alone falls short: deleting the three proxy words barely helps, and
reweighing alone only about halves the gap.

Two things the audit does not remove. Gaps of a few points either way
remain, as they do for a model fitted on the generator's ground truth. And
over-40s are shortlisted more often than under-40s (+0.042 at k = 5 and
+0.045 at k = 10 over the fresh runs), under every model including that one.
Mostly that is the jobs' minimum years: 9.1% of qualified under-40s' pairs
fall short of them in the main run, no over-40s'. The matcher prices a
shortfall more gently than the generator does (0.153 in log-odds for a year
short, against 0.4). Among candidates who meet the minimum a smaller gap is
left (+0.025 and +0.033), part of it the matcher paying for experience past
the minimum, which the ground truth never does. Trained without that input,
the matcher brings it to about what the ground-truth model shows at no
measurable ranking cost, but on one fresh run its top-5 interval for women
then falls just below zero, so by the rule fixed before those runs were read
the served matcher is unchanged. Jobs whose minimum is long for their level
are flagged to the recruiter instead (see *PDF reports*). The definitions,
every group's figures, how each configuration was chosen, and what an audit on
generated data can and cannot establish are in
[`docs/fairness_audit.md`](docs/fairness_audit.md).

## Embeddings

Every candidate and job can also be embedded by a pretrained
sentence-transformer. The evaluation above measures them as a retrieval
scorer; the API does not rank with them yet.

- **Encoder** (`app/ml/encoder.py`): all-MiniLM-L6-v2, 384 dimensions,
  pinned to one revision on the Hugging Face Hub. Its name, model id plus
  revision plus chunking, is stored beside every vector, so a change of
  weights makes old vectors visibly another encoder's. It encodes the same
  document the TF-IDF features read, folded the same way. That matters for
  this model: its tokenizer lowercases but does not apply NFKC, and measured
  on it, full-width `Python` has cosine 0.21 with `Python` as typed, and 1.0
  once folded. The model reads 256 word-pieces at most and would ignore the
  rest, so a longer document is cut at word boundaries into pieces of up to
  254 word-pieces (the model adds two markers), and its vector is the mean
  of theirs, weighted by word-piece count and rescaled to unit length. The
  name records this as `+chunks254`. Every document in the default corpus
  fits in one piece (the longest has 130), so its vector is exactly the
  model's; cutting costs 1.18 s over the 4,000 candidates, 1.4% of the 83 s
  that encoding them took in the same session (the machine was slower than
  in the runs measured below). Against "registered nurse, ICU experience", a
  680-word-piece test resume that is half data engineering and half nursing
  scores cosine 0.133 truncated and 0.411 chunked; one whose only nursing
  sentence comes after 625 word-pieces of data engineering, 0.129 and 0.171.
  The tail now counts in proportion to its length.
- **Storage** (`candidate_embeddings`, `job_embeddings`): float32 `bytea`,
  little-endian, searched exactly in numpy rather than with pgvector, which
  the local PostgreSQL does not offer. The reasoning and measurements are in
  [`docs/embedding_storage.md`](docs/embedding_storage.md). The database checks
  each vector's byte length, and a vector with NaN, infinity or all zeros is
  refused before it is written.
- **Cache** (`app/services/embeddings.py`): each row keeps the SHA-256 of the
  text it was made from, so a refresh encodes only owners that are new or
  whose text has changed. Owners go in batches, each encoded with no
  transaction open and then written and committed on its own. A batch locks
  its owners `FOR KEY SHARE`, so editing a job, or a candidate's resume or
  any other field but the email, does not wait for it. A candidate's email
  has a unique index, which makes it a key to PostgreSQL, so an edit that
  changes it does wait for a batch holding that candidate; it cannot
  deadlock with one. A vector written for text that an edit has since
  replaced keeps the old text's hash and is re-encoded next time. A delete
  waits only for the batches holding its owner at that moment, one per
  refresh. Two refreshes of the same owners finish without deadlocking,
  since both write in id order.

The encoder is an optional extra, since it brings torch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, about 550 MB installed
pip install -e ".[embeddings]"
python scripts/embed.py --out embed_report.json
```

The model's files (88 MB) are downloaded into the Hugging Face cache on first
use. CI's main job leaves the extra out, so the tests that load the real
model skip there; a separate `embeddings` job installs it and runs them,
failing rather than skipping if the model cannot be loaded. The cache and the
script are tested with a deterministic stand-in encoder.

### Measured on the default corpus

One run of `scripts/embed.py` on a scratch database holding the default corpus
(200 jobs, 4,000 candidates), CPU only, CPython 3.14.6, torch 2.14.1+cpu,
sentence-transformers 6.1.0, Windows 11:

| | Encoded | Total | In the model |
| --- | ---: | ---: | ---: |
| Candidates, first run | 4,000 | 54.57 s | 53.94 s |
| Jobs, first run | 200 | 3.27 s | 3.22 s |
| Candidates, rerun | 0 (4,000 current) | 0.11 s | 0 s |
| Candidates, after 2 resumes were edited | 2 | 0.16 s | 0.04 s |

Loading the model took 6.4 s. Encoding is the cost, at about 74 candidates a
second in the model. Everything else, reading the candidates, hashing their
text and writing the vectors, took the remaining 0.6 s. The run varies: two
later runs from scratch on the same database took 59.59 s and 62.18 s for
the candidates (65 to 68 a second in the model), 3.87 s and 3.74 s for the
jobs, and 6.98 s and 6.78 s to load the model, and a rerun with nothing to
do took 0.15 s. Reading all 4,000 back with `load_embeddings` took 57.5 ms
(median of 5; 57.7 and 58.0 ms in two later sessions), and an exact top-10
over them 2.4 ms. Five stored vectors matched a fresh encoding of the same
text exactly, though that is not guaranteed: which batch a text is encoded
in can move its vector by up to 1.6e-7 per value (measured over 64
documents).

A soak on the same database: two refreshes at once (batches of 64 and of
100) over all 4,000 candidates, while another connection deleted 50
candidates and edited 50 resumes. Both exited cleanly, and the slowest
delete took 14 ms. 3,950 vectors remained for the 3,950 candidates left,
none for a deleted one. A third run re-encoded exactly the 50 edited
candidates, and nothing was stale afterwards. A second soak added 50 email
changes, the edits that do wait for a batch. It ended the same way: both
refreshes clean, the slowest delete 10 ms and the slowest email change 2 ms,
and a third run re-encoded exactly the 50 resume edits (the email is not
part of the encoded text).

## Scoring candidates

`POST /api/v1/jobs/{id}/score` runs a pool of candidates through the trained
matcher, ranks them best first and caches one row per candidate in
`match_scores`. With no body it scores every candidate on file;
`candidate_ids` (up to 1,000) restricts the pool, and `limit` (1–500,
default 50) only trims what comes back — the whole pool is scored and cached
either way. Ranks in that response are positions within the pool just
scored, ties going to the lower candidate id. `GET /api/v1/jobs/{id}/matches`
pages through the cache, best first, without scoring anything. Its ranks run
from 1 to `total` across everything cached for the job, so they match the
scoring response whenever the whole pool was scored together.

**What a score is.** The calibrated probability that this candidate would be
advanced for this job, learned from past recruiters' decisions (`advanced`,
see the training pipeline above) after reweighing them so that gender and the
decision are independent, and without reading community lines. So it is not a
prediction of what those recruiters, with their planted bias, would have done;
the fairness audit measures how far it departs from that. Each row carries the
eight engineered features behind its score, exactly `FEATURE_NAMES`. The
TF-IDF half is not broken out per score: the calibrated matcher keeps one
fitted vocabulary per calibration fold, so there is no single text similarity
to report.

**Caching.** Rows are keyed by job, candidate and `model_version`, the first
16 hex characters of the SHA-256 of the matcher file's bytes. Scoring again
with the same matcher updates the rows in place. A retrained matcher gets a
new version on its own, and the listing shows only the current matcher's
rows; older ones stay in the table. A cached score is as of its `scored_at`:
editing the job or a candidate does not rescore it, so score again to
refresh.

**Errors.** A missing job, or any requested candidate that does not exist,
is a 404 naming it, checked before the matcher is loaded. A misspelt body
field is a 422 rather than being ignored, since ignoring it would silently
score everyone on file. A matcher that is missing or cannot be loaded is a
503 with `code: "matcher_unavailable"` and the regeneration command; the file
path and the underlying error go to the server log, not the response.

**Concurrency.** A run's write takes turns with every other write to the
job. Once its scores and breakdowns are computed, the run locks the job, as
an edit or a delete of it does, and holds the lock while it writes its
scores and commits. Another run of the same job waits for that, then updates
the first run's rows instead of racing it to insert them. A delete already
under way finishes first, and the run is then a 404; one that arrives during
the write waits for it, and so does an edit. New applications to the job do
not wait.

The wait lasts as long as the write. On 2026-09-26, with 4,000 candidates on
file, the lock was held for a median 2,031 ms by a first scoring and 1,416 ms
by a rescore, most of it the database flush (1,476 and 978 ms), and an edit
of the job sent as the lock was taken answered in 1,996–2,484 ms. The machine
was slow that day: the whole requests took a median 6.8 and 6.0 s, against
3.1 and 2.5 s measured below, so the hold is better read as a share, 30% of
a first scoring and 24% of a rescore.

So the write was shortened on 2026-09-29. The scores now go in with one
`INSERT ... ON CONFLICT DO UPDATE` per 1,000 rows, each column sent as one
array, instead of one ORM insert or update per row, and the statement's
parameters are built before the lock is taken. Rows are written in candidate
id order, whatever the ranking, so every run takes its row locks in the same
order. Timed in-process on a throwaway database with the default corpus's
4,000 candidates, the old and the new code alternating over 6 jobs:

| | Before | After |
| --- | --- | --- |
| Lock held, first scoring | 814 ms median (801–888) | 240 ms median (225–319) |
| Lock held, rescore | 598 ms median (568–640) | 224 ms median (206–253) |
| Share of the call held, first scoring | 36% | 14% |
| Share of the call held, rescore | 29% | 13% |
| Whole call (scoring and commit), first scoring | 2,278 ms median | 1,709 ms median |
| Whole call, rescore | 2,068 ms median | 1,711 ms median |

An earlier round measured a first version that still built the statement's
parameters under the lock (35% against 15%, 29% against 14%). Measured again
the same evening from exports of the two commits, on a machine then running
about three times slower, the shares held: 37% against 14% of a first scoring
(2,290 against 692 ms) and 27% against 12% of a rescore (1,664 against
631 ms). Over all three rounds, 18 jobs, the old and the new code returned
the same ranking and stored the same score, rank and features in all
72,000 rows.

Over HTTP, what changes is how long an edit of the job waits. The old and
the new code ran as two servers on one throwaway database, each scoring all
4,000 candidates for 6 jobs of its own, twice, while a client sent an edit
of the job being scored every 20 ms. The slowest edit during a run took a
median 2,577 ms against 1,065 ms for a first scoring (2,054–2,989 against
899–1,108) and 1,826 ms against 935 ms for a rescore (1,445–2,192 against
694–1,068). That was the same slow evening, and there an edit that did not
wait still took 0.3–0.5 s, since it shares the server process with the
scoring.

A candidate deleted mid-run, before the run has written its score, fails the
write. The run is then rolled back, re-read and scored again, and the
candidate has dropped out of the pool (or is a 404, if it was asked for by
id). A score someone else saved first is updated in place, which no longer
costs an attempt. Up to three attempts are made, and only a third failure is
answered with a 409 asking the client to retry.

Before the lock, some interleavings deadlocked, and one of the two requests
answered 503. A delete of the job could remove a score a run was about to
insert in a later 1,000-row batch, or one the run had just rescored; and two
runs that ranked the same new candidates differently, as an edit between
their reads can make them do, each waited on a row the other had inserted.
Each order was replayed against the scoring route, and each is now a test.
On the final code, a 100-round soak of two whole-pool runs of 1,500
candidates racing a delete of their job, and two 120 s mixed soaks from 12
and 24 threads (1,094 and 1,083 requests), answered no 5xx, and PostgreSQL
logged no deadlock.

### Measured on the default corpus

Real numbers from a live server on the development database, loaded through
the API with the default corpus's 4,000 candidates, using the matcher trained
above (CPython 3.14.6). Timings come from two runs on the same data, which
differed by about a quarter, so both are shown:

| | |
| --- | --- |
| Scoring all 4,000 for a new job | 3,521 and 2,782 ms median server time, 3 jobs per run (2,661–3,691 ms overall) |
| Scoring them again (rows updated) | 2,960 and 2,464 ms median (2,438–3,109 ms overall) |
| Listing a page of 50 | 25.8 ms median round trip over 11 requests |
| 10 simultaneous first scorings of the same 300 candidates, 5 rounds a run | 150 of 150 answered 200 over three runs, with 300 rows cached each round; with the retry disabled, 25 of 100 lost the race (409) over two runs |

A second burst raced a delete: eight requests naming the same 100
already-scored candidates and two scoring the whole pool, with one of those
candidates deleted 30 ms in, over 5 rounds. Wherever the retry was in place,
each request that named the deleted candidate answered 404 naming it. With
only two attempts, the whole-pool requests answered the retryable 409 in 9 of
20 across two runs, because the delete knocks both back and their retries
then race each other; that is why there is a third attempt, and with it all
20 succeeded across two runs. No request in any burst answered 500 or 503.

Both bursts were run again on 2026-09-26, once runs of a job took turns, on
a throwaway database with 4,000 candidates on file. Ten simultaneous first
scorings of the same 300 answered 200 in 50 of 50 over 5 rounds, with 300
rows cached each round, and still 50 of 50 with the retry disabled; the code
from before that day, with the retry disabled, lost the race in 12 of 50 the
same afternoon. In the delete burst, the 40 requests naming the deleted
candidate all answered 404 naming it and the 10 whole-pool ones 200, with
three attempts and also with two, where the older code had needed the third.

Most of the scoring time is feature computation. Under a profiler, on the
code the table above measured, normalising skill lists took about half of a
whole-pool request, and three of every four of those calls redid the job's
own lists, which do not change across the pool. The job's lists are now
normalised once per pass over the pool, and the pool is passed over once per
calibration fold and once more for the stored breakdown: 6 calls per
candidate instead of 24, with bit-identical scores and features. With an old
and a new server running side by side on the same 4,000 candidates, requests
alternating between them:

| | Before | After |
| --- | --- | --- |
| Scoring all 4,000 for a new job (3 jobs each) | 3,871 ms median | 3,061 ms median |
| Scoring them again | 3,524 ms median | 2,517 ms median |
| Scoring them again, a second round (6 each) | 3,094 ms median | 2,576 ms median |

For the same job, both servers returned the same ranks, scores and features
for the 500 best, the most one response carries. Run in-process, all 4,000
matched exactly for six jobs, and a matcher retrained on the new code was
byte-identical to the old one.

The timings above were measured on the matcher before it stopped reading
community lines and started training on reweighed pairs (2026-10-09). Run
in-process on 4,000 candidates for each of six jobs, alternating, the old
matcher scored in a median of 915 ms and the new one in 943 ms, so the
server times above should be read as about 3% low.

Timed without a profiler, in-process and without the database, skill
normalisation was 48-52% of the scoring work on the old code in three runs,
and is 17-19% on the new in six. A profiler puts the new code at 41%, because it
slows many small Python calls far more than the numerical code. Most of what
remains is the five calibration folds' TF-IDF passes over the resumes, 63-79%
of the time in three runs, against 27-32% for all the engineered features.
Absolute times varied more than threefold between runs on a busy machine, so
only the shares are given here.

Scoring everyone also no longer stops at 65,533 candidates. The cached rows
were looked up with one query parameter per candidate, and psycopg refuses
more than 65,535. Over HTTP on a throwaway database, the old code scored
65,533 candidates but answered 503 "Database unavailable" for 65,534, after
43 s of scoring. The rows are now looked up with one array parameter: the
same 65,534 scored in 54 s and rescored in 44 s, and looking up 1,000 of
them uses the unique index (2 ms). Since 2026-09-29 there is no separate
lookup: the write sends each column as one array, and a test scores and
writes 65,541 candidates in one run.

## PDF reports

Two endpoints turn a job's cached ranking into something to read and pass
on. Neither scores anything: they read the rows `POST /api/v1/jobs/{id}/score`
cached under the matcher currently on disk, so the ranks, scores and
features are the ones `GET /api/v1/jobs/{id}/matches` lists, and each report
names the matcher version and when its scores were taken.

- `GET /api/v1/jobs/{id}/report.pdf?limit=25` — the shortlist: the best
  `limit` (1–200) candidates with their current title, years, score and the
  share of the job's required and preferred skills each lists.
- `GET /api/v1/jobs/{id}/candidates/{cid}/scorecard.pdf` — one candidate's
  rank and score, the job's required and preferred skills split into listed
  and not listed, years, seniority and education against the job's, and
  the eight features behind the score with what each one means.

The skills are split with the same canonicalisation the features use
(`postgres` counts for `PostgreSQL`), and a test checks the split's shares
against the coverage features on 12,000 corpus pairs. The split is worked
out from the job and candidate as they are now, so when either was edited
after the score was taken, the scorecard says so instead of showing a score
and a breakdown that disagree.

A job with nothing cached under the current matcher, or a candidate with
no score for the job, is a 409 naming the scoring endpoint, rather than an
empty report that reads as if nobody fitted. A missing job or candidate is
a 404, and a matcher that cannot be loaded is the same 503 scoring gives.

Text is drawn in fonts embedded in each PDF, all unmodified, with their
sources, licences and checksums in
[`backend/app/assets/fonts`](backend/app/assets/fonts). Each character takes
the first that has it:

| Font | Draws |
| ---- | ----- |
| DejaVu Sans 2.37 | Latin with its accents (`Łukasz Šimić`, `Nguyễn Văn Đức`), Greek, Cyrillic, Armenian, Georgian, Hebrew, Arabic |
| Droid Sans Fallback | Chinese characters, Japanese kana (`李雷`, `陳淑芬`, `山田さくら`) |
| Nanum Square | Korean (`김민준`) |
| Noto Sans Devanagari 2.007 | Devanagari (`प्रिया कृष्णन`), shaped |
| Noto Sans Thai 2.002 | Thai (`ณัฐวุฒิ ศรีสุข`), shaped |

Names in these scripts read back from the PDF as typed, so they can be copied
and searched, with two exceptions. ReportLab lays every line out left to right
and does not join Arabic letters, so a name with Hebrew or Arabic in it is put
into the order it is seen in first (`python-bidi`), Arabic in the joined forms
its letters take (`arabic-reshaper`). It reads correctly on the page, but
copied out of the PDF those letters can come out reversed, so a search may
not find the name; and a name long enough to wrap in its table cell has its
lines in the wrong order, the end of the name on the first line.

Devanagari and Thai need shaping: letters combine into conjuncts, and vowel
signs sit before, above or below the letter they follow. A paragraph holding
either is shaped through HarfBuzz (`uharfbuzz`), and only the words in those
two fonts, so Latin beside them keeps its plain letters (no `ﬆ` or `ﬁ`
ligatures). They read correctly on the page, but a conjunct or a reph is a
glyph of its own, which the PDF maps back to a private-use character, so
copied out of the PDF such a name does not come back exactly and a search may
not find it. Without `uharfbuzz` (it is a declared dependency) ReportLab
cannot shape, so these two fonts are left out and the names are drawn as `?`
rather than drawn wrong.

Characters no embedded font has are drawn without their accents where that
leaves one a font has, and as `?` otherwise: Tamil, Ethiopic and other
scripts without a font here, and emoji (DejaVu has some, but ReportLab
cannot map characters past U+FFFF for copying). Accents
typed as separate combining marks are composed onto their letter first.
Zero-width spaces, direction marks and control characters are left out,
except that whitespace controls such as tabs and line breaks read as spaces.
Long skill lists run onto further pages.

A job whose minimum experience is above what its level usually asks for gets
a note under its line on the shortlist, marked "Check:", the same one the API
gives in the job's `requirement_notes`. A long minimum rules out candidates who
could do the job and falls hardest on younger ones; where the law protects
every age (in the UK and the EU, for instance) it has to be justified. The
usual maxima are the tops of the ranges the synthetic jobs are drawn from
(0, 1, 3, 6, 9 and 12 years from intern to principal, in
[`app/services/requirements.py`](backend/app/services/requirements.py)); a
deployment should set its own. The note informs; it changes no score.

[`docs/samples/shortlist.pdf`](docs/samples/shortlist.pdf) and
[`docs/samples/scorecard.pdf`](docs/samples/scorecard.pdf) were made by
`scripts/sample_reports.py` through a running server: one synthetic job
from a corpus seed the matcher was not trained on, its 50-candidate slate,
scored by the served matcher (`cb0b877b6182d568`). Every name, resume and
skill in them is synthetic.

```bash
python scripts/sample_reports.py --base-url http://127.0.0.1:8000/api/v1 --out docs/samples
```

## License

MIT — see [LICENSE](LICENSE).
