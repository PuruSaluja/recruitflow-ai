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
files; `--no-dense` measures everything else without them.

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
| Environment | CPython 3.14.6, scikit-learn 1.9.0; for the dense scorer, sentence-transformers 6.1.0 and torch 2.14.1+cpu |

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
  2,500 held-out scores are all distinct. Ties affect the baselines, and the
  hybrid retrieval scorer below, whose fused scores tie whenever two
  candidates' ranks are swapped between its two sides.

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

## Retrieval fitted to no label: lexical, dense and hybrid

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
  value from the paper that introduced it, not tuned. Fusion ties two
  candidates whose ranks are swapped between the sides (3rd and 7th against
  7th and 3rd); 458 of the 10,000 pairs share their fused score with another
  on their slate, and the metrics average over the order of each tie.

None of them is fitted to this corpus's labels (the embedding model was
pretrained elsewhere, and BM25 reads only the candidates' text), so besides
the 50 held-out jobs they are measured on all 200.

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
  +0.0002. On average, fusion costs dense about 0.02 to 0.04 NDCG@10.
- **Why: the two sides mostly agree, and where they differ BM25 is usually
  the worse.** Their rankings of a slate correlate closely (Spearman 0.8439
  on average over all 200 slates, 0.6517 at the lowest), so fusion has little
  independent signal to add, and equal weighting gives the weaker side as
  much say as the better one. Whether fusion helps a job follows which side
  ranks it better: on the 53 jobs where BM25 ranks at least as well as dense,
  the hybrid beats dense on 46; on the other 147, on 41. Weighting the sides,
  or another `k`, might change the overall result, but choosing either on
  these same jobs would be tuning on the test set, so neither was tried.
- **All three sit well below the trained matcher** (0.8024 on the held-out
  jobs). These scorers read only the text, while the matcher also has
  engineered features over the structured fields (skill coverage, experience,
  seniority, education) and learns their weights. Dense alone is about level
  with `required_skill_coverage` alone (0.6637 against 0.6533, intervals
  overlapping).
- Pooled ROC-AUC is in the JSON for all three, but it pools scores across
  jobs, and the hybrid's scores are fused ranks within each job's slate, so
  its pooled AUC is not comparable with the others'. The per-job NDCG and P@5
  above are the comparison that holds.

## The first stage for a re-rank

A slower, more precise scorer (a cross-encoder reads the job and the
candidate together) is too costly to run over a whole pool, so it re-orders
only the top N a cheap scorer keeps. It can only promote a candidate the
first stage let through, so a first stage is judged by **recall@N**: the
share of a slate's relevant candidates (`relevance >= 1`) in its top N,
averaged over jobs. Candidates tied with the N-th score are all kept, so the
cut never depends on the order candidates arrive in. Only the hybrid makes
such ties here: at N = 20 it keeps 20.01 candidates a slate on average over
all 200 jobs, the other two exactly 20.

The first stage and N were fixed **before any re-rank was run**, on the 150
training jobs, by a rule written beforehand
([`choose_first_stage`](../scripts/evaluate_matcher.py)): the smallest N of
5, 10, 15, 20, 25 and 30 at which some scorer keeps at least 90% of the
relevant candidates on average, and at that N the scorer that keeps the
most, the cheaper one on a tie. The held-out jobs, where a re-rank will be
measured, play no part in the choice.

| Recall@N, 150 training jobs | 5 | 10 | 15 | 20 | 25 | 30 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.3346 | 0.6086 | 0.8426 | 0.9641 | 0.9869 | 0.9911 |
| Dense | 0.3982 | 0.6710 | 0.8619 | 0.9648 | 0.9897 | 0.9920 |
| Hybrid (RRF) | 0.3984 | 0.6807 | 0.8691 | **0.9678** | 0.9896 | 0.9957 |

**Chosen: the hybrid, N = 20**, 40% of a 50-candidate slate. Read plainly:

- **At N = 20 the choice barely matters.** The three keep 96.4% to 96.8%
  of the relevant candidates on the training jobs, and on the 50 held-out
  jobs 95.3% (BM25), 95.0% (dense) and 95.0% (hybrid). The rule picks the
  highest, and the hybrid's margin over dense is 0.003.
- **The strong fits are almost all kept.** Recall of `relevance == 2` at
  N = 20 is 99.7% for the hybrid on the training jobs and 99.5% on the
  held-out ones.
- **Small N has a ceiling.** A training slate holds 8.4 relevant
  candidates on average (1 to 16), so no scorer can keep more than 64% of
  them in a top 5, or 96% in a top 10, averaged per slate. The best
  measured, 0.40 and 0.68, are well below both, so the scorers, not the
  ceiling, are what N = 10 falls short on.
- What the re-rank can gain is bounded by what the first stage keeps, and
  about 5% of the relevant candidates on a held-out slate start below the
  cut, where no second stage can reach them.

Recall on all 200 jobs and of the strong fits, and the mean number kept, are
in the JSON under `first_stage`.

## What this does not show

- **Real-world accuracy.** The ground truth is a formula over a synthetic
  profile plus noise; a real job's "fit" has no such definition.
- **Fairness.** A ranking can score well here and still treat groups
  differently: `relevance` never sees demographics, so these metrics are blind
  to the planted bias by construction. That is the fairness audit's job.
- **Tuning or model choice.** Nothing was tuned on these numbers. The
  estimators keep scikit-learn's defaults, and no other model family was
  tried. The retrieval scorers keep their textbook settings. The re-rank's
  first stage and N were chosen on the training jobs only; the cross-encoder
  re-rank itself is not measured yet.
- **Generalisation across corpora.** One corpus seed. Splits vary which jobs
  are held out, not which candidates or jobs were generated.
