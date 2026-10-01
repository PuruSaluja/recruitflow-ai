# Model evaluation

Held-out ranking quality of the matcher, measured by
[`scripts/evaluate_matcher.py`](../scripts/evaluate_matcher.py) with the
metrics in [`backend/app/ml/evaluate.py`](../backend/app/ml/evaluate.py).
Every number below is copied from
[`model_evaluation.json`](model_evaluation.json), which that script wrote in
one run; rerun it to regenerate both:

```
python scripts/train_matcher.py      # if no matcher is on disk yet
python scripts/evaluate_matcher.py
```

**All of this is measured on synthetic data.** The corpus, its ground truth
and both of its planted biases come from `app.ml.corpus` (see the README's
*Synthetic corpus* section). These numbers say how well the matcher recovers
a ground truth the generator defined; they are not evidence about real
candidates or real hiring.

## What was measured

| | |
| --- | --- |
| Matcher | `04a562c4f2c1837e` (content hash of `matcher.joblib`), trained by `scripts/train_matcher.py` on the default corpus |
| Corpus | seed 42, 200 jobs, 4,000 candidates, slates of 50 |
| Training label | `advanced`, the simulated past recruiter decision |
| Held out | 50 jobs (`--test-share 0.25 --split-seed 0`), each with its whole slate: 2,500 pairs |
| Relevant share of held-out pairs | 0.1744 (no held-out job lacks a relevant candidate) |
| Environment | CPython 3.14.6, scikit-learn 1.9.0 |

The script regenerates the corpus from the matcher's manifest, holds out the
same jobs, and first recomputes the held-out ROC-AUC the training run recorded
in that manifest (0.7829). It refuses to report anything if the two differ.

The version is a hash of the file, so it names exactly what was measured, but
a retrain from the same config can hash differently: scikit-learn's TF-IDF
vectoriser pickles `_stop_words_id`, the memory address of its stop-word
list. Retraining this config gave different bytes in exactly those places
(one per calibration fold) and bit-identical predictions on all 10,000 pairs,
so a retrained matcher reproduces every number here under a new version.

**The metrics grade against `relevance`, which the matcher never saw.** It is
fitted on `advanced`; `relevance` is the generator's ground-truth grade of a
pair (2 strong, 1 partial, 0 not a fit), and ranking the people who truly fit
first is what the matcher is for.

- **NDCG@10**: gain `2^relevance - 1` (a strong fit is worth three partial
  ones), divided by the best DCG the job's own slate allows.
- **P@5**: share of the top five who are at least a partial fit.
- Both are per job over its 50-candidate slate, then averaged over jobs.
  Intervals are 95% percentile bootstraps over the 50 held-out jobs (2,000
  resamples).
- **ROC-AUC** is pooled over all 2,500 held-out pairs.
- Tied scores are averaged over every order of the tie, which makes "all
  scores equal" the exact expected value of a random ranking. The matcher's
  2,500 held-out scores are all distinct, so this only affects the baselines.

## Results on the held-out jobs

| Scorer | NDCG@10 | P@5 | ROC-AUC vs `advanced` | ROC-AUC vs `relevance >= 1` |
| --- | --- | --- | --- | --- |
| **Matcher** | **0.8024** (0.7668–0.8353) | **0.7760** (0.7160–0.8280) | **0.7829** | **0.9427** |
| Random ranking (exact expectation) | 0.1638 | 0.1744 | 0.5000 | 0.5000 |
| `required_skill_coverage` alone, untrained | 0.6533 (0.6064–0.6974) | 0.6419 (0.5753–0.7022) | 0.7480 | 0.8826 |
| Same pipeline fitted on `relevance >= 1` (reference only) | 0.8382 (0.8068–0.8676) | 0.8360 (0.7840–0.8840) | 0.7922 | 0.9567 |

Against `relevance == 2` (strong fits only) the matcher's pooled ROC-AUC is
0.9429.

Read plainly:

- **The matcher ranks far above chance.** About 3.9 of its top five
  candidates per job are relevant on average (P@5 0.7760), against 0.87 for a
  random order on the same slates.
- **The model adds a good deal over required-skill coverage alone.**
  Ranking by that one feature gets NDCG@10 0.6533; the trained matcher gets
  0.8024, and the two intervals do not overlap. It was chosen up front as the
  obvious recruiter heuristic and is the only single feature in the JSON. A
  one-off check during review, which the script does not repeat, ranked these
  jobs by each of the nine named features alone: this one scored highest on
  NDCG@10 and P@5, with skill Jaccard close behind.
- **Learning from past decisions costs something, on this corpus.** The same
  pipeline fitted on the ground truth instead scores 0.8382 NDCG@10 and 0.8360
  P@5, 0.036 and 0.060 above the matcher. Its intervals overlap the matcher's,
  and this is one split with no paired test, so the size of the gap is not
  pinned down. It is also not a fair competitor: no real system has
  ground-truth labels, which is why the matcher is not trained that way (see
  `app/ml/train.py`). The gap mixes two causes, the noise of any recruiter
  decision and the planted bias against women in `advanced`; this evaluation
  does not separate them. The fairness audit is where the bias is measured.

## Spread across splits

Which 50 jobs land in the held-out set moves these numbers. The script
retrains the same pipeline, on the same label, on nine more splits of the same
corpus (`--split-seed` 1 to 9) and measures each on its own held-out jobs:

| Over 10 splits | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| NDCG@10 | 0.8063 | 0.0231 | 0.7662 | 0.8521 |
| P@5 | 0.8012 | 0.0227 | 0.7520 | 0.8280 |
| ROC-AUC vs `advanced` | 0.7836 | 0.0082 | 0.7698 | 0.7941 |
| ROC-AUC vs `relevance >= 1` | 0.9498 | 0.0034 | 0.9427 | 0.9549 |

The persisted matcher's split (seed 0) is typical for NDCG@10 and somewhat
below the mean for P@5 (0.7760 against 0.8012); it also has the lowest ROC-AUC
against `relevance` of the ten. The headline figures above are that one
matcher's, since it is the one the API serves.

## What this does not show

- **Real-world accuracy.** The ground truth is a formula over a synthetic
  profile plus noise; a real job's "fit" has no such definition.
- **Fairness.** A ranking can score well here and still treat groups
  differently: `relevance` never sees demographics, so these metrics are blind
  to the planted bias by construction. That is the fairness audit's job.
- **Tuning or model choice.** Nothing was tuned on these numbers. The
  estimators keep scikit-learn's defaults, and no other model family was
  tried. Dense embeddings, hybrid retrieval and a cross-encoder re-rank are
  planned, to be measured on these same metrics and held-out jobs.
- **Generalisation across corpora.** One corpus seed. Splits vary which jobs
  are held out, not which candidates or jobs were generated.
