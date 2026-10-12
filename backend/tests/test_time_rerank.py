"""Unit tests for scripts/time_rerank.py, the per-stage latency CLI. No database needed.

Every test uses the stand-in encoder and cross-encoder; what is checked is what
gets timed and how the timings are summarised, not how fast anything is.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from fake_cross_encoder import FakeCrossEncoder
from fake_encoder import FakeEncoder

from app.config import REPO_ROOT
from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.embeddings import cosine_scores
from app.ml.encoder import EncoderUnavailable, candidate_document, job_document
from app.ml.retrieval import BM25Index, first_stage_cut, reciprocal_rank_fusion
from app.ml.train import training_data

SCRIPT = REPO_ROOT / "scripts" / "time_rerank.py"
CONFIG = CorpusConfig(seed=5, n_jobs=40, n_candidates=1200)
N = 5


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load(SCRIPT, "time_rerank")


def _evaluation(first_stage: str = "hybrid_rrf", **rerank: Any) -> dict[str, Any]:
    return {
        "corpus_config": dataclasses.asdict(CONFIG),
        "retrieval": {"encoder": "fake-encoder@1"},
        "rerank": {
            "cross_encoder": "fake-cross-encoder@1",
            "first_stage": first_stage,
            "n": N,
            **rerank,
        },
    }


class _Recorder:
    """Hands out one stand-in of each kind and keeps them to look at."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.encoders: list[FakeEncoder] = []
        self.cross_encoders: list[FakeCrossEncoder] = []
        monkeypatch.setattr(script, "load_encoder", self._encoder)
        monkeypatch.setattr(script, "load_cross_encoder", self._cross_encoder)

    def _encoder(self) -> FakeEncoder:
        self.encoders.append(FakeEncoder())
        return self.encoders[-1]

    def _cross_encoder(self) -> FakeCrossEncoder:
        self.cross_encoders.append(FakeCrossEncoder())
        return self.cross_encoders[-1]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    return _Recorder(monkeypatch)


def _ticking(step: float = 0.002) -> Callable[[], float]:
    """A clock that moves ``step`` seconds each time it is read."""
    now = [0.0]

    def clock() -> float:
        now[0] += step
        return now[0]

    return clock


@pytest.fixture(scope="module")
def slates() -> list[dict[str, Any]]:
    """Every slate, with its kept candidates, worked out independently."""
    corpus = generate_corpus(CONFIG)
    data = training_data(corpus)
    index = BM25Index([candidate_document(c) for c in corpus.candidates])
    rows = {c.ref: i for i, c in enumerate(corpus.candidates)}
    jobs = {j.ref: j for j in corpus.jobs}
    out = []
    for ref in sorted(set(data.job_refs)):
        candidates = [cand for (job, cand) in data.pairs if job.ref == ref]
        job = job_document(jobs[ref])
        documents = [candidate_document(c) for c in candidates]
        lexical = index.scores(job)[[rows[c.ref] for c in candidates]]
        encoder = FakeEncoder()
        dense = cosine_scores(encoder.encode(documents), encoder.encode([job])[0])
        out.append(
            {
                "job": job,
                "documents": documents,
                "kept": {
                    "hybrid_rrf": first_stage_cut(reciprocal_rank_fusion(lexical, dense), N),
                    "bm25": first_stage_cut(lexical, N),
                    "dense": first_stage_cut(dense, N),
                },
            }
        )
    return out


# --- what is timed -----------------------------------------------------


def test_each_stage_of_each_slate_is_timed_once_per_repeat(recorder: _Recorder) -> None:
    results = script.run(_evaluation(), repeats=2, jobs=3, clock=_ticking())
    assert len(results["repeats"]) == 2
    for timed in results["repeats"]:
        assert list(timed) == list(script.STAGES)
        # Each timing reads the clock twice, 2 ms apart.
        for stage in script.STAGES:
            assert timed[stage] == {"median_ms": 2.0, "p90_ms": 2.0, "mean_ms": 2.0}
    assert results["median_ms_across_repeats"]["cross_encoder_kept"] == {"min": 2.0, "max": 2.0}
    assert (results["slates"], results["slate_size_mean"], results["pool"]) == (3, 50.0, 1200)
    assert (results["first_stage"], results["n"]) == ("hybrid_rrf", N)
    assert (results["encoder"], results["cross_encoder"]) == (
        "fake-encoder@1",
        "fake-cross-encoder@1",
    )


def test_the_cross_encoder_sees_the_kept_then_the_whole_slate(
    recorder: _Recorder, slates: list[dict[str, Any]]
) -> None:
    results = script.run(_evaluation(), repeats=2, jobs=3, clock=_ticking())
    [cross_encoder] = recorder.cross_encoders
    expected = []
    for slate in slates[:3]:
        kept = [
            d for d, k in zip(slate["documents"], slate["kept"]["hybrid_rrf"], strict=True) if k
        ]
        expected += [(slate["job"], kept), (slate["job"], slate["documents"])]
    # A warm-up of the first slate, then every slate in each repeat.
    assert cross_encoder.calls == expected[:2] + expected + expected
    kept_sizes = [int(s["kept"]["hybrid_rrf"].sum()) for s in slates[:3]]
    assert results["kept_mean"] == pytest.approx(np.mean(kept_sizes))
    assert min(kept_sizes) >= N


def test_a_timing_is_the_clock_between_its_two_reads(recorder: _Recorder) -> None:
    durations = iter([0.001, 0.003] * 1000)
    now = [0.0]

    def clock() -> float:
        now[0] += next(durations)
        return now[0]

    # The clock is read twice a timing: every timing lasts 3 ms after a 1 ms gap.
    results = script.run(_evaluation(), repeats=1, jobs=3, clock=clock)
    assert results["repeats"][0]["dense"]["median_ms"] == 3.0


def test_percentiles_are_over_the_slates(recorder: _Recorder) -> None:
    assert script._stats([0.001, 0.002, 0.003, 0.004, 0.010]) == {
        "median_ms": 3.0,
        "p90_ms": 7.6,
        "mean_ms": 4.0,
    }


@pytest.mark.parametrize(("first_stage", "left_out"), [("bm25", "dense"), ("dense", "bm25")])
def test_only_the_steps_the_first_stage_needs_are_timed(
    recorder: _Recorder, slates: list[dict[str, Any]], first_stage: str, left_out: str
) -> None:
    results = script.run(_evaluation(first_stage), repeats=1, jobs=3, clock=_ticking())
    assert left_out not in results["repeats"][0] and first_stage in results["repeats"][0]
    assert (results["encoder"] is None) == (first_stage == "bm25")
    assert len(recorder.encoders) == (first_stage == "dense")
    kept = [int(s["kept"][first_stage].sum()) for s in slates[:3]]
    assert results["kept_mean"] == pytest.approx(np.mean(kept))


def test_all_jobs_by_default(recorder: _Recorder, slates: list[dict[str, Any]]) -> None:
    results = script.run(_evaluation(), repeats=1, jobs=None, clock=_ticking())
    assert results["slates"] == 40
    # Fusion ties at the cut on some slates, so more than N are kept on average.
    kept = [int(s["kept"]["hybrid_rrf"].sum()) for s in slates]
    assert results["kept_mean"] == pytest.approx(np.mean(kept)) and np.mean(kept) > N


def test_the_spread_is_the_lowest_and_highest_repeats_median(recorder: _Recorder) -> None:
    # Each reading moves this clock 1 microsecond further than the last did, so
    # each repeat's timings run longer than the one before.
    step, now = [0.0], [0.0]

    def slowing() -> float:
        step[0] += 0.000001
        now[0] += step[0]
        return now[0]

    results = script.run(_evaluation(), repeats=3, jobs=2, clock=slowing)
    for stage, spread in results["median_ms_across_repeats"].items():
        medians = [timed[stage]["median_ms"] for timed in results["repeats"]]
        assert medians == sorted(medians) and medians[0] < medians[-1]
        assert spread == {"min": medians[0], "max": medians[-1]}


# --- refusals ----------------------------------------------------------


@pytest.mark.parametrize(
    ("evaluation", "message"),
    [
        ({**_evaluation(), "rerank": None}, "no re-rank to time"),
        (_evaluation("matcher"), "unknown first stage 'matcher'"),
        ({**_evaluation(), "corpus_config": None}, "no corpus_config"),
        ({**_evaluation(), "retrieval": {"encoder": "other@2"}}, "ran encoder 'other@2'"),
        (_evaluation(cross_encoder="other@3"), "ran cross-encoder 'other@3'"),
    ],
)
def test_an_evaluation_it_cannot_time_is_refused(
    recorder: _Recorder, evaluation: dict, message: str
) -> None:
    with pytest.raises(script.TimingRefused, match=message):
        script.run(evaluation, repeats=1, jobs=1)


@pytest.mark.parametrize(
    ("evaluation", "message"),
    [
        ([1, 2], "not a JSON object"),
        ({**_evaluation(), "retrieval": None}, "no retrieval object"),
        ({**_evaluation(), "rerank": {"first_stage": "bm25", "cross_encoder": "x"}}, "n = None"),
        (_evaluation(n=0), "n = 0"),
        (_evaluation(n="20"), "n = '20'"),
        (_evaluation(n=True), "n = True"),
        (_evaluation(n=2.5), "n = 2.5"),
        ({**_evaluation(), "corpus_config": {"seed": 5, "bogus": 1}}, "corpus_config cannot be"),
        ({**_evaluation(), "corpus_config": {"n_jobs": 0}}, "corpus_config cannot be"),
    ],
)
def test_a_damaged_evaluation_is_refused_before_any_model_loads(
    recorder: _Recorder, evaluation: object, message: str
) -> None:
    with pytest.raises(script.TimingRefused, match=message):
        script.run(evaluation, repeats=1, jobs=1)
    assert recorder.encoders == [] and recorder.cross_encoders == []


def test_a_damaged_evaluation_file_exits_2_naming_the_problem(
    recorder: _Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _main(tmp_path, _evaluation(n=0))
    assert code == 2 and "n = 0" in capsys.readouterr().err
    assert not out.exists()


def _main(tmp_path: Path, evaluation: dict | None, *extra: str) -> tuple[int, Path]:
    source, out = tmp_path / "evaluation.json", tmp_path / "out" / "latency.json"
    if evaluation is not None:
        source.write_text(json.dumps(evaluation), encoding="utf-8")
    code = script.main(["--evaluation", str(source), "--out", str(out), *extra])
    return code, out


def test_it_writes_the_json_and_prints_each_stage(
    recorder: _Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _main(tmp_path, _evaluation(), "--repeats", "1", "--jobs", "2")
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["slates"] == 2 and len(written["repeats"]) == 1
    printed = capsys.readouterr().out
    assert all(stage in printed for stage in script.STAGES)


def test_an_out_path_it_cannot_write_exits_2_naming_it(
    recorder: _Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "out").write_text("a file where the folder should be", encoding="utf-8")

    code, out = _main(tmp_path, _evaluation(), "--repeats", "1", "--jobs", "1")

    assert code == 2
    assert f"cannot write {out}" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [(["--repeats", "0"], "--repeats must be at least 1"), (["--jobs", "0"], "--jobs must be")],
)
def test_bad_counts_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    code, out = _main(tmp_path, _evaluation(), *extra)
    assert code == 2 and message in capsys.readouterr().err and not out.exists()


def test_a_missing_or_damaged_evaluation_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(tmp_path, None)[0] == 2
    (tmp_path / "evaluation.json").write_text("{not json", encoding="utf-8")
    code = script.main(["--evaluation", str(tmp_path / "evaluation.json")])
    assert code == 2 and capsys.readouterr().err.count("cannot read") == 2


def test_a_refusal_or_a_model_it_cannot_load_exits_2(
    recorder: _Recorder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _main(tmp_path, {**_evaluation(), "rerank": None})[0] == 2
    assert "no re-rank to time" in capsys.readouterr().err

    def unavailable() -> None:
        raise EncoderUnavailable("cannot load the cross-encoder")

    monkeypatch.setattr(script, "load_cross_encoder", unavailable)
    code, out = _main(tmp_path, _evaluation())
    assert code == 2 and "cannot load the cross-encoder" in capsys.readouterr().err
    assert not out.exists()


def test_an_out_path_it_cannot_write_is_refused_before_the_timings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "out").write_text("a file where the folder should be", encoding="utf-8")
    monkeypatch.setattr(script, "run", lambda *a, **k: pytest.fail("the timings ran"))

    code, out = _main(tmp_path, _evaluation(), "--repeats", "1", "--jobs", "1")

    assert code == 2
    assert f"cannot write {out}" in capsys.readouterr().err
