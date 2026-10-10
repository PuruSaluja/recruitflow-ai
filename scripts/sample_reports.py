"""Make the sample shortlist and scorecard PDFs through a running API.

Usage::

    python scripts/sample_reports.py --base-url http://127.0.0.1:8000/api/v1 --out docs/samples

Generates a small synthetic corpus, creates one of its jobs and that job's
slate of candidates through the API, scores the slate with the matcher the
server has loaded, and saves what the two report endpoints return: the
shortlist, and the scorecard of the candidate ranked first. Everything goes
through HTTP, as a client would use it, so point it at a server whose
database can take the new rows: the job and candidates are left in place.

Every name, resume and skill in the samples is synthetic. Needs the ``dev``
extra, for its HTTP client (``httpx2``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx2

from app.ml.corpus import CorpusConfig, generate_corpus

#: Not the corpus the served matcher was trained on (seed 42), so the sample
#: job is one it has never seen.
SAMPLE_CORPUS = CorpusConfig(seed=2026, n_jobs=4, n_candidates=1000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--out", type=Path, default=Path("docs/samples"))
    parser.add_argument("--limit", type=int, default=15, help="rows on the sample shortlist")
    return parser


def _ok(response: httpx2.Response, expected: int = 200) -> httpx2.Response:
    if response.status_code != expected:
        raise RuntimeError(
            f"{response.request.method} {response.request.url} gave "
            f"{response.status_code}: {response.text[:300]}"
        )
    return response


def run(client: httpx2.Client, out: Path, limit: int) -> list[Path]:
    """Create the sample job and slate through ``client``, score them and save both reports."""
    corpus = generate_corpus(SAMPLE_CORPUS)
    job = corpus.jobs[0]
    slate = {pair.candidate_ref for pair in corpus.pairs if pair.job_ref == job.ref}
    people = [c for c in corpus.candidates if c.ref in slate]

    job_id = _ok(client.post("/jobs", json=job.api_payload()), 201).json()["id"]
    ids = [
        _ok(client.post("/candidates", json=person.api_payload()), 201).json()["id"]
        for person in people
    ]
    scored = _ok(
        client.post(f"/jobs/{job_id}/score", json={"candidate_ids": ids, "limit": 1})
    ).json()
    top = scored["results"][0]["candidate"]["id"]

    out.mkdir(parents=True, exist_ok=True)
    shortlist = out / "shortlist.pdf"
    scorecard = out / "scorecard.pdf"
    shortlist.write_bytes(
        _ok(client.get(f"/jobs/{job_id}/report.pdf", params={"limit": limit})).content
    )
    scorecard.write_bytes(_ok(client.get(f"/jobs/{job_id}/candidates/{top}/scorecard.pdf")).content)
    print(
        f"job {job_id} ({job.title}): {scored['scored_count']} candidates scored by matcher "
        f"{scored['model_version']}; wrote {shortlist} and {scorecard}"
    )
    return [shortlist, scorecard]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with httpx2.Client(base_url=args.base_url, timeout=120) as client:
            run(client, args.out, args.limit)
    except (httpx2.HTTPError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
