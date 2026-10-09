"""Training the matcher: the held-out split, the pipeline and the calibrated classifier.

**The training target is** ``advanced`` **-- whether a past recruiter moved the
candidate forward -- and not** ``relevance``. That is the decision this module
records, and it is deliberate. ``relevance`` is ground truth derived from a
candidate's ``true_skills``, which no real recruiting system has: a production
matcher is trained on the decisions its company actually made. Training on
``relevance`` would also make the fairness audit vacuous, because the generated
bias lives entirely in ``advanced`` -- a model fitted on ground truth has
nothing unfair to find. So the matcher learns from the biased record on
purpose, and ``relevance`` is carried alongside for ranking evaluation and
never fitted.

**What the bias would do is measured, then mitigated.** Trained plainly on
that record, the matcher shortlisted qualified women 5.0 points less often
than qualified men on average over nine audit runs, mostly through a
women-in-tech community line some women's resumes carry
(``docs/fairness_audit.md``). So the served matcher does two things:

* it never reads a line headed with one of :data:`EXCLUDED_HEADINGS`: a
  community or affiliation line says nothing a job asks for, and it is where
  the proxy lives;
* :func:`train_matcher` reweighs the training pairs (Kamiran and Calders) so
  that gender and the past decision are independent in what the classifier
  learns from. Gender is read only to compute those weights, from the
  training corpus; it is never a feature, and scoring never needs it.

With both, no audit run showed a gap for women or for nonbinary candidates
against men whose interval excluded zero, and no run could tell the cost in
NDCG@10 from zero. Over-40s are shortlisted more often than under-40s on some
corpora, by about as much as under a model fitted on ground truth: younger
candidates more often fall short of a job's minimum years, which the ground
truth counts against them.

The split is **by job**: every pair on a job's slate lands on the same side.
Ranking metrics are per job over a whole slate, so a slate torn between train
and test cannot be scored at all. The cost is that a candidate may sit on one
slate in training and another in test -- 84.5% of test candidates do, on the
default corpus. Measured, that overlap does not inflate the held-out score: on
pairs whose job *and* candidate were both held out, a model trained the strict
way scores 0.8022 ROC-AUC and this job-split model scores 0.8030 on the same
pairs, while the strict split would throw away 25% of the training pairs.

The classifier is logistic regression: the matrix is sparse and mostly TF-IDF
columns, its coefficients name the term they weigh (which is what the fairness
audit needs to point at), and it emits the probabilities calibration expects.
Nothing here is tuned -- the estimators keep scikit-learn's defaults.
``scripts/evaluate_matcher.py`` measures this matcher against baselines on its
held-out jobs; no other model family has been compared.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from app.ml.corpus import Corpus
from app.ml.matrix import Pair, PairFeaturiser
from app.ml.mitigation import LineDroppingFeaturiser, WeightedPipeline, reweighing_weights

#: The corpus field the matcher is fitted on. See this module's docstring.
TRAINING_LABEL = "advanced"

#: Share of jobs held out by default.
DEFAULT_TEST_SHARE = 0.25

#: How probabilities are calibrated. Sigmoid and isotonic calibrate this corpus
#: equally well (Brier 0.1071 against 0.1068, log loss 0.3539 against 0.3528),
#: but sigmoid is strictly monotone, so calibrating cannot reorder a slate or
#: tie two candidates together. Isotonic is a step function: on the same 2,500
#: held-out pairs it returned 786 distinct probabilities against sigmoid's
#: 2,500, and every tie is a pair of candidates a ranking has to separate
#: arbitrarily.
CALIBRATION_METHOD = "sigmoid"

#: Lines the served matcher never reads, matched at the start of a line after
#: folding. Only the heading the synthetic corpus writes is listed, because only
#: it was measured; real resumes would need their own list.
EXCLUDED_HEADINGS = ("Community:",)

#: The candidate attribute the training pairs are reweighed against.
REWEIGHED_ATTRIBUTE = "gender"


@dataclass(frozen=True, eq=False)
class TrainingData:
    """Pairs with their labels, in one order, ready to fit or to score.

    Compared and hashed by identity. The generated field-by-field ``==``
    compared the label arrays, which have no single truth value, so two equal
    instances raised ``ValueError`` and none could be hashed.
    """

    pairs: tuple[Pair, ...]
    #: The training target: whether a past recruiter advanced the candidate.
    advanced: np.ndarray
    #: The ground-truth grade, carried for ranking evaluation and never fitted.
    relevance: np.ndarray
    #: Which job each row belongs to; the key the split groups on.
    job_refs: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.pairs)


def training_data(corpus: Corpus) -> TrainingData:
    """Resolve a corpus's labelled pairs into objects and parallel label arrays."""
    jobs = {job.ref: job for job in corpus.jobs}
    candidates = {candidate.ref: candidate for candidate in corpus.candidates}
    return TrainingData(
        pairs=tuple((jobs[pair.job_ref], candidates[pair.candidate_ref]) for pair in corpus.pairs),
        advanced=np.array([pair.advanced for pair in corpus.pairs], dtype=bool),
        relevance=np.array([pair.relevance for pair in corpus.pairs], dtype=int),
        job_refs=tuple(pair.job_ref for pair in corpus.pairs),
    )


def split_by_job(
    data: TrainingData, *, test_share: float = DEFAULT_TEST_SHARE, seed: int = 0
) -> tuple[TrainingData, TrainingData]:
    """Hold out ``test_share`` of the jobs, with each job's whole slate.

    Jobs are drawn from a sorted list, so the same seed always holds out the
    same jobs whatever order the pairs arrive in.
    """
    refs = sorted(set(data.job_refs))
    n_test = round(len(refs) * test_share)
    if not 0 < n_test < len(refs):
        raise ValueError(
            f"A test share of {test_share} takes {n_test} of {len(refs)} jobs; it must leave at "
            "least one job on each side."
        )
    shuffled = list(refs)
    random.Random(seed).shuffle(shuffled)
    held_out = set(shuffled[:n_test])
    is_test = np.array([ref in held_out for ref in data.job_refs])
    return _subset(data, ~is_test), _subset(data, is_test)


def _subset(data: TrainingData, mask: np.ndarray) -> TrainingData:
    return TrainingData(
        pairs=tuple(pair for pair, keep in zip(data.pairs, mask, strict=True) if keep),
        advanced=data.advanced[mask],
        relevance=data.relevance[mask],
        job_refs=tuple(ref for ref, keep in zip(data.job_refs, mask, strict=True) if keep),
    )


def make_matcher(featuriser: PairFeaturiser | None = None) -> CalibratedClassifierCV:
    """An unfitted matcher: featuriser, classifier, and calibration around both.

    By default the featuriser skips :data:`EXCLUDED_HEADINGS`; the fairness
    audit passes a plain :class:`PairFeaturiser` to measure the matcher
    without it. ``fit`` accepts ``sample_weight``, which reaches the classifier
    and the calibration alike.

    Calibration wraps the whole pipeline rather than sitting inside it, so each
    of its folds learns its own vocabulary from its own training part. A
    featuriser fitted once outside the folds would have read every resume the
    calibration is then fitted against.
    """
    if featuriser is None:
        featuriser = LineDroppingFeaturiser(EXCLUDED_HEADINGS)
    pipeline = WeightedPipeline([("features", featuriser), ("classifier", LogisticRegression())])
    return CalibratedClassifierCV(pipeline, method=CALIBRATION_METHOD)


def training_weights(data: TrainingData) -> np.ndarray:
    """Reweighing weights for ``data``: :data:`REWEIGHED_ATTRIBUTE` against the target."""
    groups = [getattr(candidate, REWEIGHED_ATTRIBUTE) for _, candidate in data.pairs]
    return reweighing_weights(groups, getattr(data, TRAINING_LABEL))


def train_matcher(data: TrainingData) -> CalibratedClassifierCV:
    """Fit the served matcher on ``data``. The only place a training target is chosen."""
    return make_matcher().fit(
        data.pairs, getattr(data, TRAINING_LABEL), sample_weight=training_weights(data)
    )
