# Embedding storage

**Decision (2026-10-01): no pgvector for now.** Resume and job embeddings
will be stored in PostgreSQL as `bytea` holding float32 values, and compared
by exact cosine similarity in numpy over a job's candidate pool.

The numbers below come from
[`scripts/spike_vector_storage.py`](../scripts/spike_vector_storage.py), which
wrote [`embedding_storage.json`](embedding_storage.json) in one run; rerun it
to regenerate them:

```
python scripts/spike_vector_storage.py --out docs/embedding_storage.json
```

## Can pgvector be installed here?

Not on the local server as it stands:

- The local PostgreSQL 16.10 is a binaries-only Visual C++ build, not a
  service. `vector` is not among its `pg_available_extensions` (the script
  checks this on whatever server it is pointed at).
- pgvector's README gives Windows installation as a build from source:
  Visual Studio with C++ support, then `nmake` in the x64 Native Tools
  command prompt. This machine has no Visual Studio, no Windows SDK and no C
  compiler. Installing the Visual Studio build tools is a large install that
  was not made as part of a build session.
- The README names two other routes. Docker: Docker Desktop is installed but
  its engine was not running, and a container would be a second PostgreSQL
  server rather than this one, for development and tests alike.
  conda-forge (`conda install -c conda-forge pgvector`): it does publish a
  Windows build (0.8.6 for win-64), made against conda-forge's own
  PostgreSQL, pinned below 17 on Windows, so the same major version as this
  server. Whether its library loads into this separately built server is
  untested, and nothing was downloaded.

Whichever server runs the tests needs the extension too, so CI's service
container would have to change with it.

## The fallback, measured

Random unit vectors, 384 dimensions (all-MiniLM-L6-v2's size), stored in a
temporary table twice: as `real[]` and as `bytea` holding the raw float32
bytes. Each read fetches every vector in id order and builds one float32
matrix; the search is an exact cosine top-10 in numpy. Medians of 5 rounds,
read forms in rotating order. PostgreSQL 16.10 on localhost, CPython 3.14.6,
numpy 2.4.6, psycopg 3.3.5, Windows 11.

| Vectors | `real[]`, text | `real[]`, binary | `bytea` | Top-10 search |
| ---: | ---: | ---: | ---: | ---: |
| 4,000 | 1,063.2 ms | 449.7 ms | 88.8 ms | 3.1 ms |
| 20,000 | 5,667.1 ms | 2,669.5 ms | 655.5 ms | 16.4 ms |

Every read returned the stored float32 values bit for bit, in all three
forms; the script refuses to report timings otherwise.

The times vary from run to run. Two reruns the same evening, not committed
(one as above, one at 4,000 only with another seed and 9 rounds), read the
vectors back 15-36% faster than the run above, with the search within 10%
of it; for 4,000 vectors, text `real[]` took about 15 times as long as
`bytea` rather than 12. The order of the three forms, and so the decision,
was the same in all three runs; the exact figures were not.

What this shows:

- **The search is not the cost.** An exact top-10 over 4,000 vectors took
  3.1 ms, and 16.4 ms over 20,000. What pgvector's approximate indexes buy
  is speed at sizes far beyond this corpus, so at this scale exact search
  loses nothing.
- **Reading the vectors back is the cost, and the storage form decides it.**
  For 4,000 vectors, `real[]` read in psycopg's default text format took
  about 12 times as long as `bytea`, and binary format about 5 times; in both,
  psycopg hands back a list of Python floats that numpy then has to convert.
  Set beside the README's latest medians of 2,517-3,061 ms to score 4,000
  candidates (measured on another day, so only a rough comparison), text
  `real[]` would add 35-42% and `bytea` 3-4%.
- **Storage size is not a differentiator.** Per vector, `real[]` took 1,556
  bytes and `bytea` 1,540 (the raw values are 1,536); pgvector's docs give
  `4 * dimensions + 8`, 1,544 bytes, for its own type.
- Loading took 5.6 s for 4,000 and 29.5 s for 20,000, one measurement each,
  but that is one `COPY` writing both columns in text format, so it says
  little about writing `bytea` alone and played no part in the choice.

## What the embeddings item should do

- Store each embedding as `bytea` with an explicit byte order: write
  `vector.astype("<f4").tobytes()` and read with `np.frombuffer(..., "<f4")`,
  so the bytes mean the same on any machine. Check the length in SQL
  (`octet_length = 4 * dimensions`), as the spike's table does.
- Store the encoder's name and version beside each embedding, the way
  `match_scores` carries a model version, so a change of encoder makes old
  embeddings visibly stale rather than silently mixed.
- `np.frombuffer` returns a read-only view of the fetched bytes; the spike's
  own test tripped on that. Copy before normalising in place.
- Search exactly in numpy over the job's pool.

The costs of this choice: the vectors are opaque in SQL (hex in `psql`, no
similarity queries in SQL), and the byte order is a convention the code has
to keep. Revisit pgvector if similarity has to run inside SQL, or if pools
grow far past the sizes measured here. The cheapest route to try first is
the conda-forge package, checked against this server before anything
depends on it; either it or the Visual Studio build needs an install to be
approved first.
