# Fairness audit

Who reaches the top of a slate, by gender and by age band, measured by
[`scripts/audit_fairness.py`](../scripts/audit_fairness.py) with the metrics
in [`backend/app/ml/fairness.py`](../backend/app/ml/fairness.py) and the
mitigations in [`backend/app/ml/mitigation.py`](../backend/app/ml/mitigation.py).
Every number below is copied from one of two JSON files the script wrote:
[`fairness_audit.json`](fairness_audit.json), the nine runs the served
matcher's configuration was chosen on, and
[`fairness_audit_fresh.json`](fairness_audit_fresh.json), eight runs that
played no part in the choice. A second run of the first wrote the same JSON.
There are four exceptions, each marked where it appears: the single-split
figure in *Why out of fold*, the share of women's resumes carrying the planted
line, the fit of qualified men and women in *Where the women's gap comes
from*, and the word weights in *How the configuration was chosen*. To
regenerate both (over an hour each on a laptop, most of it the counterfactual
scoring in *Over-40s and the job's minimum years*):

```
python scripts/audit_fairness.py
python scripts/audit_fairness.py --fold-seed 5 --fold-seeds 6 7 8 --corpus-seeds 5 6 7 8 --out docs/fairness_audit_fresh.json
```

**All of this is measured on synthetic data.** The candidates, their gender
and age band, the ground truth and both planted biases come from
`app.ml.corpus` (see the README's *Synthetic corpus* section). The audit can
show whether it finds a bias of known origin and whether a mitigation removes
it. It says nothing about bias in real hiring. The last section says what it
can and cannot establish.

## The result

The past decisions the matcher learns from are biased against women, as
planted. Trained plainly on them, the matcher carried part of that bias into
its shortlists: qualified women made a top 5 4.6 to 5.0 points less often
than qualified men, averaged over runs, and 6 of 8 fresh runs' intervals
excluded zero. **The matcher the app serves never reads a resume's community
line and is trained on reweighed pairs.** With both:

| Women against men, equal-opportunity difference | Top 5 | Top 10 |
| --- | --- | --- |
| Served matcher, 9 runs it was chosen on: mean (runs whose interval excludes zero) | −0.011 (0 of 9) | −0.018 (0 of 9) |
| Served matcher, 8 fresh runs | −0.013 (0 of 8) | −0.010 (0 of 8) |
| Without the mitigation, 9 runs | −0.050 (5 of 9) | −0.046 (6 of 9) |
| Without the mitigation, 8 fresh runs | −0.046 (6 of 8) | −0.040 (6 of 8) |

No interval for women, at either cut-off, excludes zero in any of the 17
runs, and none for nonbinary candidates does in the 8 fresh runs. Ranking
quality is unchanged: in no run is the served matcher's NDCG@10 measurably
lower than the unmitigated one's.

"No gap that can be told from zero" is the claim, not "no gap". Even a model
fitted on the generator's ground truth, which learns from no biased label at
all, shows gaps of several points either way from corpus to corpus. The
served matcher's figures sit in the same range as that model's. Over-40s are
shortlisted more often than under-40s in some runs, under every model
including that one; *Other groups* explains why.

## What was measured

| | |
| --- | --- |
| Corpus | seed 42, 200 jobs, 4,000 candidates, slates of 50: 10,000 pairs |
| Pairs by gender | 5,559 men, 3,909 women, 532 nonbinary |
| Pairs by age band | 8,982 under 40, 1,018 aged 40 and over |
| Scores | out of fold: the 200 jobs dealt into 4 folds, each scored by a matcher trained on the other three |
| Shortlist | each slate's top 5, and its top 10 |
| Intervals | 95% percentile bootstrap over jobs, 2,000 resamples |
| Runs the configuration was chosen on | the main run (fold seed 0), 4 other fold seeds of this corpus (1–4) and 4 other corpora (seeds 1–4, fold seed 0) |
| Fresh runs | this corpus on fold seeds 5–8, and corpora 5–8 on fold seed 5 |
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

**Why out of fold.** One train/test split holds out only 50 jobs. On those
(the persisted matcher's split, before the mitigation), a separate check found
women's ratio at k = 5 to be 0.991 with an interval of 0.78 to 1.26, too wide
to say anything. Scoring every job with a matcher that never saw its labels
quadruples the pairs. It audits the training procedure, of which the
persisted matcher is one run.

**Why repeat it.** The intervals cover which jobs were sampled, not which
jobs each model was trained on, nor which corpus the generator drew. Both
move the figures: the fold seed alone moves the unmitigated matcher's top-5
equal-opportunity difference for women between −0.057 and −0.078 on this
corpus. So the sections below report the main run (corpus seed 42, fold seed
0) in detail, and the conclusions rest on the repeats.

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

The same bias is planted in every corpus, but the four-fifths rule does not
always see it. Across the five corpora the record's ratio runs from 0.708 to
0.821, and it passes the rule on two of them (0.814 and 0.821). The rule
compares raw selection rates, so chance in who is qualified and in the
simulated decisions moves it several points either side of 0.75. The
equal-opportunity difference finds the bias on all five corpora: −0.125 to
−0.149, every interval excluding zero.

## The matcher without the mitigation

The same classifier, trained on `advanced` with every line of every resume
read and no weights:

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

Its NDCG@10 over all 200 jobs is 0.8039 (0.7869–0.8213). At k = 5 qualified
women make the shortlist 6.3 points less often than qualified men (44.0%
against 50.4%), and the interval excludes zero. Over the nine runs its top-5
difference is negative every time, from −0.017 to −0.078; the interval
excludes zero on all five splits of this corpus but on none of the four other
corpora, so this corpus is a harsher draw than most.

## Where the women's gap comes from

The matcher never sees gender. It can only lower women's scores through
something it reads, and the corpus plants one such thing: a women-in-tech
community line on a quarter of women's resumes (24.9%, as the README's
corpus summary measures) and on no one else's. Three measurements separate it
from the rest:

| Top 5, women against men | Ratio | Equal-opportunity difference | NDCG@10 |
| --- | --- | --- | --- |
| Without the mitigation | 0.828 | −0.063 (−0.107 to −0.021) | 0.8039 |
| Same matchers, every planted line swapped for its neutral one | 0.932 | −0.010 (−0.057 to 0.034) | |
| Trained on the same corpus without the historical bias (reference) | 0.891 | −0.030 (−0.078 to 0.017) | 0.8080 |
| Fitted on ground truth (reference; no real system has it) | 0.933 | −0.026 (−0.069 to 0.017) | 0.8313 |

**The swap is the causal measurement.** Generating the corpus with
`proxy_share=0` gives the same people, slates and labels; the script checks
this. The only difference is that each planted line is replaced by the
neutral community line the same person would otherwise have written. The
same out-of-fold matchers score both versions. Every one of the 1,017 pairs
carrying a planted line (363 candidates) scores lower with it, by 0.028 on
average (median 0.020). No other pair's score moves at all. Those pairs make
a top 5 9.1% of the time with the neutral line and 5.2% with the planted one.
Swapping the line accounts for most of the women's gap: the equal-opportunity
difference goes from −0.063 to −0.010 in this run, and from a mean of −0.050
to a mean of −0.012 over the nine runs. The served matchers, scored the same
way, do not move at all: the largest change in any served score is 0.0.

The two references bracket what is left. On all five splits of this corpus,
the pipeline fitted on ground truth shows −0.026 to −0.033, though it learns
from no biased label; on the other four corpora it shows −0.031 to +0.044.
So about −0.03 of women's gap on this corpus belongs to the corpus, not to
the matcher or its training labels. A likely reason, from a separate check
not in the JSON: this corpus's qualified women are slightly weaker fits than
its qualified men (mean ground-truth fit 0.559 against 0.563; 36.6% strong
against 37.5%). On the two corpora where qualified women are clearly the
stronger (seeds 1 and 2), the reference's difference is positive. The
relation is loose, though: corpus 3 has weaker qualified women and a positive
difference. So this is an explanation that fits, not one that was shown.

## The served matcher

`app.ml.train.train_matcher` skips every line headed `Community:` before
vectorising, and weights each training pair so that, in the weighted data,
gender and the past decision are independent (Kamiran and Calders, 2012). It
reads each training candidate's gender only to compute those weights; gender
is never a feature, and scoring a candidate never needs it. In the main run:

| Top k | Group | Selection rate | Ratio | Four-fifths | Equal-opportunity difference |
| --- | --- | --- | --- | --- | --- |
| 5 | Men | 10.3% | 1 | | 0 |
| 5 | Women | 9.6% | 0.939 (0.827–1.054) | passes | −0.005 (−0.051 to 0.040) |
| 5 | Nonbinary | 9.8% | 0.952 (0.720–1.198) | passes | −0.060 (−0.158 to 0.036) |
| 5 | 40 and over | 12.5% | 1.284 (1.037–1.556) | passes | +0.021 (−0.049 to 0.093) |
| 10 | Men | 20.4% | 1 | | 0 |
| 10 | Women | 19.2% | 0.943 (0.869–1.024) | passes | −0.007 (−0.048 to 0.033) |
| 10 | Nonbinary | 21.4% | 1.050 (0.883–1.226) | passes | −0.025 (−0.108 to 0.059) |
| 10 | 40 and over | 23.2% | 1.180 (0.997–1.365) | passes | +0.031 (−0.033 to 0.090) |

Its NDCG@10 is 0.8037 (0.7857–0.8213). Job by job it differs from the
unmitigated matcher's by −0.0003 (−0.0071 to 0.0065), better on 80 jobs,
equal on 42 and worse on 78. Every interval in the table includes zero, and
every ratio passes the four-fifths rule.

## How the configuration was chosen

Four configurations were compared, each one or two changes to the unmitigated
matcher and each audited on the same nine runs:

- **Drop the proxy terms**: leave `women`, `bootcamp` and `member` out of the
  vocabulary. These are the only words that occur nowhere but in the planted
  lines; a test checks this.
- **Drop the community line**: remove every line headed `Community:` before
  vectorising, whatever it says.
- **Reweighing** alone, with every line read.
- **Both the line drop and reweighing**: the served matcher.

| Women against men, over the nine runs | Top 5: mean | Runs excluding zero | Top 10: mean | Runs excluding zero |
| --- | --- | --- | --- | --- |
| Without the mitigation | −0.050 | 5 | −0.046 | 6 |
| Drop the proxy terms | −0.043 | 5 | −0.035 | 4 |
| Drop the community line | −0.014 | 0 | −0.021 | 1 |
| Reweighing | −0.028 | 3 | −0.023 | 2 |
| Both (served) | −0.011 | 0 | −0.018 | 0 |
| Every planted line swapped for its neutral one | −0.012 | 0 | −0.020 | 1 |

The rule was fixed before looking at any fresh run: serve the configuration
with no women's interval excluding zero at either cut-off in any of the nine
runs, and accept it only if that held on fresh runs too. Only the
combination met it. Dropping the line alone came closest, missing once at
k = 10 (corpus 4: −0.046, interval −0.085 to −0.004). On the eight fresh runs the
combination held (see *Checked on fresh runs*).

**Deleting the proxy words is the weakest fix.** It is the obvious one, and
it removes every word the planted lines have to themselves, yet it leaves a
gap whose interval excludes zero in as many runs as no mitigation at all. The
planted lines' other words appear elsewhere too: "organiser", "meetup",
"mentor" and "coding" in the neutral community lines, and "data",
"engineering", "network" and "in" in titles, duties and degrees. Once some
women's lines use them, they carry a little of the signal. A separate check,
not in the JSON, shows this. It fits the classifier on all 10,000 pairs with
the three words dropped, once on this corpus and once on the same corpus with
no planted lines, and compares the weights. All 12 other words of the planted
lines get a lower weight, for example `coding` −0.25 against +0.17,
`network` −0.85 against −0.27 and `organiser` −0.05 against +0.10. All 12
words that only the neutral lines use get a higher one, for example `club`
+0.41 against −0.03 and `chess` +0.17 against −0.21. A word list fixes only
the words someone thought to list.

**Dropping the whole line does about as well as never having planted it**:
over the nine runs it averages −0.014 at k = 5, against the swap's −0.012.
On this corpus that is easy, because every such line starts with the same
heading. Real resumes rarely label an interests or community section so
reliably, and their names, email addresses and addresses carry the same kind
of signal. The generated resumes carry no contact details, so that could not
be measured here.

**Reweighing alone only about halves the gap.** It acts on the label rather
than the text, so it works without knowing what carries the bias and would
also act on proxies nobody has found. But with the line still read, its
interval excludes zero in 3 of the nine runs at k = 5. Combined with the line
drop, it removes the one remaining miss at k = 10.

**No configuration costs ranking quality.** Over the nine runs no
configuration's NDCG@10 is measurably below the unmitigated matcher's in any
run. The served one's per-job difference averages +0.0023 and is measurably
*better* in one run.

## Checked on fresh runs

Eight more runs, none of which played any part in the choice. For the served
matcher, the unmitigated one and the reference fitted on ground truth:

| Equal-opportunity difference over the 8 fresh runs: mean (runs excluding zero) | Served | Without the mitigation | Fitted on ground truth |
| --- | --- | --- | --- |
| Women, top 5 | −0.013 (0) | −0.046 (6 below) | −0.010 (0) |
| Women, top 10 | −0.010 (0) | −0.040 (6 below) | +0.001 (1 above) |
| Nonbinary, top 5 | −0.035 (0) | −0.024 (0) | −0.032 (1 above) |
| Nonbinary, top 10 | −0.021 (0) | −0.015 (0) | −0.033 (0) |
| 40 and over, top 5 | +0.042 (1 above) | +0.049 (1 above) | +0.036 (1 above) |
| 40 and over, top 10 | +0.045 (2 above) | +0.041 (2 above) | +0.050 (3 above) |

On fresh data the served matcher shows no interval excluding zero for women
or nonbinary candidates at either cut-off, where the unmitigated matcher
misses for women in 6 of 8 runs at each. Its NDCG@10 is never measurably
different from the unmitigated matcher's (per-job mean +0.0019).

## Other groups

**Nonbinary candidates** are 532 of the 10,000 pairs, about 100 of them
qualified in a run, so their intervals are wide (in the main run, −0.158 to
+0.036 at k = 5). Over the nine runs the served matcher's top-10 difference
averages −0.038 and excludes zero in one run. Every other model shows about
the same: −0.042 for the pipeline fitted on ground truth, −0.045 without the
mitigation. Gender is drawn independently of every qualification, and five of
the nine runs share one corpus, so this reads as the corpora's small
nonbinary group rather than anything the matcher learned. On the eight fresh
runs no nonbinary interval excludes zero.

**Over-40s are shortlisted more often** than under-40s, under every model,
the one fitted on ground truth included. The next section takes it apart.

## Over-40s and the job's minimum years

Equal-opportunity difference for qualified over-40s against under-40s,
averaged over runs (in brackets, runs whose interval is above zero; none is
below):

| | 9 runs, k = 5 | k = 10 | 8 fresh runs, k = 5 | k = 10 |
| --- | --- | --- | --- | --- |
| Served matcher, every qualified pair | +0.053 (2) | +0.057 (5) | +0.042 (1) | +0.045 (2) |
| Served matcher, pairs meeting the job's minimum | +0.037 (1) | +0.044 (1) | +0.025 (1) | +0.033 (2) |
| Fitted on ground truth, every qualified pair | +0.036 (2) | +0.040 (1) | +0.036 (1) | +0.050 (3) |
| Fitted on ground truth, meeting the minimum | +0.012 (0) | +0.019 (1) | +0.014 (1) | +0.031 (1) |
| Served without `experience_surplus`, meeting the minimum | +0.016 (0) | +0.026 (0) | +0.006 (1) | +0.018 (2) |

**Most of it is the jobs' minimum years.** In the main run, 9.1% of the
1,487 qualified pairs with an under-40 fall short of the job's minimum, and
none of the 212 with an over-40; over-40s who meet it go 7.4 years past it on
average, under-40s 2.8 (`experience` in the JSON). The generator's ground
truth counts a shortfall against a candidate, and so does every model, so
"qualified" over-40s are more often the stronger fits. Removing that part
would mean ranking candidates who fall short of a job's stated minimum above
ones who meet it.

**The matcher prices a shortfall more gently than the generator does.**
With everything else on the profile held fixed, and the resume's stated
years rewritten to match, a candidate 1 year short of the minimum scores
0.153 lower in log-odds than one exactly at it, 3 years short 0.421 lower and
5 years short 0.669 lower. The generator's ground truth takes 0.4, 1.2 and
2.0 (`matcher_price_of_years`, main run, the 3,000 pairs whose job asks for 5
years or more). One year short does not cost as much as five, and the
matcher is, if anything, more lenient than the rule it learns from.

**The requirement does not explain all of it.** Among qualified candidates
who meet the minimum, the served matcher still favours over-40s, by more than
the model fitted on ground truth does. Part of what is left is real in the
generator: a candidate's age follows their education's starting age plus
their years, so older candidates more often hold the education a job asks
for, which the ground truth rewards. Part is the matcher's own: it pays for
experience past the minimum, +0.157 in log-odds at 3 years past, +0.210 at 5
and +0.215 at 10 (the feature stops at 5), which the ground truth never does.
Over-40s have more of it.

**Without that input.** The served configuration trained without the
`experience_surplus` feature (`served_without_experience_surplus`) closes
the gap among candidates who meet the minimum to about what the ground-truth
model shows, at an NDCG@10 cost of 0.0004 on the nine runs (0.8055 against
0.8059) and 0.0011 on the fresh ones (0.8069 against 0.8080). It was chosen on
corpora 42 and 1 to 4 before any fresh run was read, with a rule fixed then:
adopt it only if, on the fresh runs, its mean gap among candidates meeting the
minimum is below the served matcher's and no more than 0.01 above the
ground-truth model's at both cut-offs, its mean NDCG@10 is less than 0.005
lower, and no run's interval for women excludes zero. The first two held. The
third did not: on corpus 6, its top-5 interval for women runs from −0.107 to
−0.006, where the served matcher's runs from −0.103 to +0.0003. So the served
matcher is unchanged, and whether to adopt this configuration anyway is left
as a decision.

**The requirement itself** is where an age gap enters hiring. Where the law
protects every age (in the UK and the EU, for instance), a long minimum has
to be justified. The API flags a job whose minimum is above what its level
usually asks for (`requirement_notes`), and its shortlist prints the note.

## What this audit can and cannot establish

It can establish that the audit works. It recovers the planted historical
bias at its known size (0.750 for 0.25). It finds that a model trained on the
biased record carries part of it into its shortlists, and traces most of that
to the one feature the generator planted for the purpose. And it shows, over
17 runs on nine generated corpora, that the served configuration leaves no
gap for women that can be told from zero, at no measurable cost in ranking.

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
  in for a protected attribute, and none is in this corpus. Skipping a
  `Community:` line only works because the generator labels it.
- **Reweighing needs gender at training time.** The synthetic corpus has it.
  A real deployment would need demographic data for its training set, with
  the legal and consent questions that raises.
- **The groups are coarse and few.** There are three genders and two age
  bands, with no race or ethnicity, no disability, and no intersections.
- **Many comparisons are made.** The main run reports 71 equal-opportunity
  intervals for groups other than the reference. If there were no gap at
  all, three or four would exclude zero by chance. Five do. Three are women's
  gaps with a known cause: the record, and at k = 5 the unmitigated matcher
  and the one with the proxy terms dropped. The other two have no mechanism in
  the generator and are best read as that noise: nonbinary candidates at
  k = 5 under the ground-truth pipeline (−0.125), and over-40s at k = 10
  under the unbiased-record matcher (+0.063).
- **The four-fifths rule is a screening heuristic.** Passing it is not a
  finding of fairness, and failing it is not a finding of discrimination.
  Here it missed a planted 25% bias in the record on two of five corpora.
- **The intervals are per run.** They cover which jobs were sampled, not
  which jobs a model was trained on or which corpus was drawn. The repeats
  show those move the figures by more than a single run's interval suggests.
