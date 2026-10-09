# Fairness audit

Who reaches the top of a slate, by gender and by age band, measured by
[`scripts/audit_fairness.py`](../scripts/audit_fairness.py) with the metrics
in [`backend/app/ml/fairness.py`](../backend/app/ml/fairness.py) and the
mitigations in [`backend/app/ml/mitigation.py`](../backend/app/ml/mitigation.py).
Every number below is copied from [`fairness_audit.json`](fairness_audit.json),
which that script wrote in one run (a second run wrote the same JSON). The
one exception, the word weights in *Mitigations*, is marked as a separate
check. To regenerate the JSON:

```
python scripts/audit_fairness.py
```

**All of this is measured on synthetic data.** The candidates, their gender
and age band, the ground truth and both planted biases come from
`app.ml.corpus` (see the README's *Synthetic corpus* section). The audit can
show whether it finds a bias of known origin and whether a mitigation removes
it. It says nothing about bias in real hiring. The last section says what it
can and cannot establish.

## What was measured

| | |
| --- | --- |
| Corpus | seed 42, 200 jobs, 4,000 candidates, slates of 50: 10,000 pairs |
| Pairs by gender | 5,559 men, 3,909 women, 532 nonbinary |
| Pairs by age band | 8,982 under 40, 1,018 aged 40 and over |
| Scores | out of fold: the 200 jobs dealt into 4 folds, each scored by a matcher trained on the other three |
| Shortlist | each slate's top 5, and its top 10 |
| Intervals | 95% percentile bootstrap over jobs, 2,000 resamples |
| Environment | CPython 3.14.6, scikit-learn 1.9.0, numpy 2.4.6 |

The matcher does not accept or reject anyone. It orders a slate and a
recruiter reads the top of it, so **being selected means being in a slate's
top k**. Ties at the cut are shared out, as in the ranking metrics. Three
figures are computed per group:

- **Selection rate**: the share of a group's pairs that make the shortlist.
- **Disparate impact ratio**: a group's selection rate over the reference
  group's (men; under 40). US employment guidance treats a ratio below 0.8 as
  evidence of adverse impact. This is the *four-fifths rule*, a rule of thumb
  for a first look rather than a legal test, and it says nothing about why
  two rates differ.
- **Equal-opportunity difference**: a group's true-positive rate minus the
  reference group's. The true-positive rate is the share of the group's
  qualified pairs (relevance 1 or 2, from the generator's ground truth) that
  make the shortlist. A negative value means qualified members of the group
  are shortlisted less often than qualified members of the reference group.

The two can disagree, and on this corpus they do. Selection rates differ
whenever one group is more often qualified. On these slates 17.3% of men's
pairs are qualified, against 16.3% of women's, 19.5% of nonbinary
candidates' and 20.8% of over-40s'. Gender is drawn independently of every
qualification, so the gap between men and women is chance in which candidates
landed on which slates. Age band is not independent: it follows years of
experience. So a ratio somewhat off 1 is expected even from a fair ranker, and
the equal-opportunity difference is the fairer read.

**Why out of fold.** One train/test split holds out only 50 jobs. On those,
women's ratio at k = 5 came out at 0.991 with an interval of 0.79 to 1.25, too
wide to say anything. Scoring every job with a matcher that never saw its
labels quadruples the pairs. It audits the training procedure, of which the
persisted matcher is one run.

## The record the matcher learns from

The matcher is trained on `advanced`, the simulated past recruiter's decisions
(see *Training pipeline* in the README). Read as a decision on every pair:

| Group | Advanced | Ratio to reference | Qualified pairs advanced | Equal-opportunity difference |
| --- | --- | --- | --- | --- |
| Men | 16.5% | 1 | 57.5% | 0 |
| Women | 12.4% | **0.750** (0.679–0.827) | 42.9% | **−0.145** (−0.198 to −0.092) |
| Nonbinary | 19.9% | 1.204 (0.988–1.429) | 58.7% | +0.012 (−0.088 to 0.110) |
| Aged 40 and over (against under 40) | 17.3% | 1.163 (0.977–1.359) | 54.2% | +0.025 (−0.045 to 0.099) |

The record fails the four-fifths rule for women. This is the planted
`historical_bias` of 0.25 recovered: 0.750 against the 0.75 it implies.
Qualified women were advanced 14.5 points less often than qualified men.

## The matcher

| Top k | Group | Selection rate | Ratio | Four-fifths | Equal-opportunity difference |
| --- | --- | --- | --- | --- | --- |
| 5 | Men | 10.7% | 1 | | 0 |
| 5 | Women | 8.9% | **0.828** (0.734–0.929) | passes | **−0.063** (−0.107 to −0.021) |
| 5 | Nonbinary | 10.7% | 0.999 (0.748–1.258) | passes | −0.032 (−0.133 to 0.068) |
| 5 | 40 and over | 11.8% | 1.203 (0.973–1.456) | passes | −0.007 (−0.074 to 0.062) |
| 10 | Men | 21.0% | 1 | | 0 |
| 10 | Women | 18.3% | **0.871** (0.803–0.945) | passes | −0.038 (−0.080 to 0.005) |
| 10 | Nonbinary | 22.2% | 1.057 (0.893–1.226) | passes | −0.033 (−0.112 to 0.046) |
| 10 | 40 and over | 22.1% | 1.118 (0.940–1.295) | passes | +0.013 (−0.048 to 0.072) |

Its NDCG@10 over all 200 jobs is 0.8039 (0.7869–0.8213).

The gap is smaller than the record's, though a decision on every pair and a
place in a top k are not directly comparable, and it is not closed. Women's ratio
passes the four-fifths rule at both cut-offs. At k = 5, though, the interval
reaches down to 0.734, so this corpus cannot rule out a ratio below 0.8. The
clearer finding is the equal-opportunity difference: at k = 5, qualified
women make the shortlist 6.3 points less often than qualified men (44.0%
against 50.4%), and the interval excludes zero. At k = 10 the gap is smaller
and its interval just reaches zero.

Neither the nonbinary group nor the over-40 group shows a gap whose interval
excludes zero. The nonbinary group is small (104 qualified pairs), so its
intervals are wide: at k = 5 the equal-opportunity difference could be
anywhere from −0.133 to +0.068.

## Where the women's gap comes from

The matcher never sees gender. It can only lower women's scores through
something it reads, and the corpus plants one such thing: a women-in-tech
community line on a quarter of women's resumes (24.9%, as the README's
corpus summary measures) and on no one else's. Three measurements separate it
from the rest:

| Top 5, women against men | Ratio | Equal-opportunity difference | NDCG@10 |
| --- | --- | --- | --- |
| Matcher | 0.828 | −0.063 (−0.107 to −0.021) | 0.8039 |
| Same matchers, every planted line swapped for its neutral one | 0.932 | −0.010 (−0.057 to 0.034) | |
| Matcher trained on the same corpus without the historical bias (reference) | 0.891 | −0.030 (−0.078 to 0.017) | 0.8080 |
| Pipeline fitted on ground truth (reference; no real system has it) | 0.933 | −0.026 (−0.069 to 0.017) | 0.8313 |

**The swap is the causal measurement.** Generating the corpus with
`proxy_share=0` gives the same people, slates and labels; the script checks
this. The only difference is that each planted line is replaced by the
neutral community line the same person would otherwise have written. The
same out-of-fold matchers score both versions. Every one of the 1,017 pairs
carrying a planted line (363 candidates) scores lower with it, by 0.028 on
average (median 0.020). No other pair's score moves at all. Those pairs make
a top 5 9.1% of the time with the neutral line and 5.2% with the planted one.
Swapping the line accounts for most of the women's gap: the equal-opportunity
difference goes from −0.063 to −0.010.

The two references bracket what is left. Trained without the historical bias,
the same pipeline still shows −0.030, and fitted on ground truth it shows
−0.026. Neither interval excludes zero. A residual of that size is what chance
in the slates produces on this corpus, not a second mechanism.

## Mitigations

Four ways to train the matcher differently, each audited the same way and
compared with the matcher job by job on NDCG@10. None of them is used by the
matcher the app serves.

- **Drop the proxy terms**: leave `women`, `bootcamp` and `member` out of the
  vocabulary. These are the only words that occur nowhere but in the planted
  lines; a test checks this.
- **Drop the community line**: remove every line headed `Community:` before
  vectorising, whatever it says.
- **Reweighing** (Kamiran and Calders, 2012): weight each training pair so
  that, in the weighted data, gender and the past decision are independent.
  It needs each training candidate's gender to compute the weights. The
  model never sees gender as a feature, and scoring does not need it.
- **Drop the community line and reweigh.**

| Top 5, women against men | Ratio | Equal-opportunity difference | NDCG@10 | NDCG@10 against the matcher, per job |
| --- | --- | --- | --- | --- |
| Matcher | 0.828 (0.734–0.929) | −0.063 (−0.107 to −0.021) | 0.8039 | |
| Drop the proxy terms | 0.848 (0.753–0.955) | −0.051 (−0.095 to −0.008) | 0.8021 | −0.0019 (−0.0068 to 0.0018) |
| Drop the community line | 0.938 (0.833–1.052) | −0.012 (−0.058 to 0.031) | 0.8029 | −0.0010 (−0.0068 to 0.0046) |
| Reweighing | 0.896 (0.793–1.009) | −0.023 (−0.066 to 0.021) | 0.8010 | −0.0030 (−0.0092 to 0.0025) |
| Drop the line and reweigh | 0.939 (0.827–1.054) | −0.005 (−0.051 to 0.040) | 0.8037 | −0.0003 (−0.0071 to 0.0065) |

At k = 10 the order is the same. Women's equal-opportunity difference is
−0.0378 for the matcher, −0.0221 with the terms dropped, −0.0095 with the
line dropped, −0.0184 with reweighing and −0.0074 with both.

**Deleting the proxy words is the weakest fix.** It is the obvious one, and
it removes every word the planted lines have to themselves. Yet the gap at
k = 5 only shrinks from −0.063 to −0.051, and its interval still excludes
zero. The planted lines' other words appear elsewhere too: "organiser",
"meetup", "mentor" and "coding" in the neutral community lines, and "data",
"engineering", "network" and "in" in titles, duties and degrees. Once some
women's lines use them, they carry a little of the signal.

A separate check, not in the JSON, shows this. It fits the classifier on all
10,000 pairs with the three words dropped, once on this corpus and once on
the same corpus with no planted lines, and compares the weights. Every other
word of the planted lines gets a lower weight: `coding` −0.25 against +0.17,
`network` −0.85 against −0.27, `organiser` −0.05 against +0.10. Every word
that only the neutral lines use gets a higher one: `club` +0.41 against
−0.03, `chess` +0.17 against −0.21. A word list fixes only the words someone
thought to list.

**Dropping the whole line does about as well as never having planted it.** It
gives 0.938 and −0.012, against 0.932 and −0.010 for the swap, at no
measurable cost in ranking. On this corpus that is easy, because every such
line starts with the same heading. Real resumes rarely label an interests or
community section so reliably, and their names, email addresses and
addresses carry the same kind of signal. The generated resumes carry no
contact details, so that could not be measured here.

**Reweighing acts on the label instead of the text.** It brings the gap to
−0.023 without knowing what carries the bias. That makes it the one
mitigation here that would also act on proxies nobody has found. Combined
with dropping the line, it gives the smallest gap measured, −0.005.

**No mitigation measurably costs ranking quality.** No mean per-job NDCG@10
difference from the matcher is larger than 0.003, and every interval includes
zero.
Women's ratio passes the four-fifths rule under every mitigation, but every
interval still reaches below 0.85. On this corpus, 200 jobs cannot pin a
ratio down to better than about ±0.1.

## What this audit can and cannot establish

It can establish that the audit works. It recovers the planted historical
bias at its known size (0.750 for 0.25). It finds that a model trained on the
biased record carries part of it into its shortlists. And it traces most of
that to the one feature the generator planted for the purpose. It also shows,
on this corpus, which mitigations remove the gap and that they cost no
measurable ranking quality.

It cannot establish anything about real candidates or real hiring:

- **The groups are generated.** Gender is drawn independently of every
  qualification. In real data, group membership correlates with education,
  experience and career breaks in ways no generator reproduces. A ratio
  below 1 would then mix bias with real differences in the qualified pool.
- **The ground truth is the generator's.** The equal-opportunity difference
  here is computed against a `relevance` that real systems never have.
  Measured against a real outcome, such as later job performance, it would
  inherit whatever bias that outcome carries.
- **The proxy is planted, and there is only one.** Real resumes carry names,
  email domains, addresses, gaps, hobbies and phrasing. Any of them can stand
  in for a protected attribute, and none is in this corpus.
- **The groups are coarse and few.** There are three genders and two age
  bands, with no race or ethnicity, no disability, and no intersections. The
  nonbinary group is 5% of the pool, and its intervals span about 20 points.
- **Many comparisons are made.** The run reports 49 equal-opportunity
  intervals for groups other than the reference. If there were no gap at
  all, two or three would exclude zero by chance. Five do. Three are women's
  gaps with a known cause: the record, the matcher at k = 5, and the matcher
  with the proxy terms dropped at k = 5. The other two have no mechanism in
  the generator and are best read as that noise: nonbinary candidates at
  k = 5 under the ground-truth pipeline (−0.125), and over-40s at k = 10
  under the unbiased-record matcher (+0.063).
- **The four-fifths rule is a screening heuristic.** Passing it is not a
  finding of fairness, and failing it is not a finding of discrimination.
