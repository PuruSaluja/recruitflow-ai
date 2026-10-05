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

The dense and hybrid scorers need the `embeddings` extra and the model's
files; `--no-dense` measures everything else without them. The run that wrote
the JSON took 3 min 33 s on CPU, nine retrains included.

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

## Retrieval without training: lexical, dense and hybrid

Three more scorers rank the same slates without fitting anything to a label
([`backend/app/ml/retrieval.py`](../backend/app/ml/retrieval.py)):

- **BM25**: the job's document (title, description, both skill lists) is the
  query, searched against every candidate's document (current title, resume,
  listed skills) over the corpus's pool of 4,000. Words are folded and
  tokenised as the TF-IDF features do. `k1 = 1.2`, `b = 0.75`, the textbook
  defaults, not tuned.
- **Dense**: cosine between the two documents' embeddings from
  all-MiniLM-L6-v2, pinned to revision `1110a243`.
- **Hybrid**: reciprocal rank fusion of the two, `sum(1 / (60 + rank))`,
  ranked within each job's slate, both sides weighted equally. `k = 60` is the
  value from the paper that introduced it, not tuned.

None of them is trained, so every job is unseen to them, and besides the 50
held-out jobs they are measured on all 200, four times the sample.

| Scorer | 50 held-out jobs: NDCG@10 | P@5 | All 200 jobs: NDCG@10 | P@5 |
| --- | --- | --- | --- | --- |
| BM25 | 0.5081 (0.4635–0.5533) | 0.5000 (0.4320–0.5680) | 0.5333 (0.5097–0.5574) | 0.5330 (0.5010–0.5650) |
| Dense | 0.6637 (0.6170–0.7096) | 0.6760 (0.6120–0.7360) | 0.6566 (0.6313–0.6828) | 0.6660 (0.6310–0.7010) |
| **Hybrid (RRF)** | **0.6259** (0.5790–0.6689) | **0.6200** (0.5560–0.6760) | **0.6392** (0.6156–0.6637) | **0.6415** (0.6100–0.6735) |

**Does fusion beat either side alone?** Compared job by job on the same
slates, with a 95% bootstrap interval over jobs for the mean paired
difference:

| Hybrid minus | Jobs | NDCG@10 | P@5 | Hybrid better / tied / worse (NDCG@10) |
| --- | --- | --- | --- | --- |
| BM25 | 50 held out | +0.1178 (+0.0834 to +0.1549) | +0.1200 (+0.0680 to +0.1720) | 43 / 0 / 7 |
| BM25 | all 200 | +0.1059 (+0.0886 to +0.1231) | +0.1085 (+0.0850 to +0.1320) | 158 / 2 / 40 |
| Dense | 50 held out | −0.0378 (−0.0655 to −0.0096) | −0.0560 (−0.1001 to −0.0120) | 17 / 1 / 32 |
| Dense | all 200 | −0.0174 (−0.0344 to +0.0002) | −0.0245 (−0.0475 to −0.0005) | 87 / 2 / 111 |

Read plainly:

- **No: the hybrid does not beat dense retrieval on this corpus.** It beats
  BM25 clearly, on both metrics and in about four jobs of five, but it ranks
  slightly *worse* than dense alone: lower on both metrics and on both sets of
  jobs, worse on more jobs than it is better. Three of the four intervals
  against dense exclude zero; the fourth, NDCG@10 over all 200 jobs, reaches
  +0.0002. Here fusion costs dense about 0.02 to 0.04 NDCG@10 and gains
  nothing.
- **A likely reason is the gap between the two sides.** BM25 is far weaker
  than dense on these documents (0.53 against 0.66 NDCG@10 over all jobs), and
  equal-weight fusion gives its ranking as much say as the better one's.
  Fusion tends to help when the two sides are comparably good and wrong about
  different candidates. Weighting the sides, or another `k`, might change
  that, but choosing either on these same jobs would be tuning on the test
  set, so neither was tried.
- **All three sit well below the trained matcher** (0.8024 on the held-out
  jobs). Untrained retrieval reads only the text, while the matcher also has
  engineered features over the structured fields (skill coverage, experience,
  seniority, education) and learns their weights. Dense alone is about level
  with `required_skill_coverage` alone (0.6637 against 0.6533, intervals
  overlapping).
- Pooled ROC-AUC is in the JSON for all three, but it pools scores across
  jobs, and the hybrid's scores are fused ranks within each job's slate, so
  its pooled AUC is not comparable with the others'. The per-job NDCG and P@5
  above are the comparison that holds.

## What this does not show

- **Real-world accuracy.** The ground truth is a formula over a synthetic
  profile plus noise; a real job's "fit" has no such definition.
- **Fairness.** A ranking can score well here and still treat groups
  differently: `relevance` never sees demographics, so these metrics are blind
  to the planted bias by construction. That is the fairness audit's job.
- **Tuning or model choice.** Nothing was tuned on these numbers. The
  estimators keep scikit-learn's defaults, and no other model family was
  tried. The retrieval scorers keep their textbook settings. A cross-encoder
  re-rank is planned, to be measured on these same metrics and held-out jobs.
- **Generalisation across corpora.** One corpus seed. Splits vary which jobs
  are held out, not which candidates or jobs were generated.
