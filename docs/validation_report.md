# Day 4 — Validation Report

**Synthetic lateral-movement data: does it substitute for scarce real attack data?**

104 tests · **v1, restated 2026-09-07** on the leak-free fit-window graph (§7.4)

> **This report was rewritten.** Every headline number in the version dated 2026-08-11 was produced
> on a graph built from the whole 58-day corpus, including the evaluation window. The verdict on
> synthetic augmentation survived; the numbers supporting it did not, and two secondary conclusions
> reversed outright. §7.4 has the mechanism, the cost, and why a `t_hi` guard that existed and was
> tested still failed to prevent it. Process write-up: [`v1_process.md`](v1_process.md).

---

## What this measures, and what it does not

The detector in this report is a **measuring instrument for synthetic-data quality**, not a
detection-science contribution. Nothing here claims a new way to find lateral movement. Every
number exists to answer one question: *is this synthetic data useful?*

The answer is **no**. The evidence is clean; it is not tidy, and the difference matters — see §1 on
what the leak was smoothing over.

All numbers are AUC-PR (average precision) measured on the **real holdout**, never on synthetic
data. The holdout is **51 real red-team events / 6 campaigns against 205,612 benign events** — a
base rate of **0.000248 (1:4,032)**. Chance is 0.000248.

That base rate is a **sampling choice, not a deployment property** — the negatives are a draw from
the holdout window, which itself holds on the order of 149 million events (§5). Every *absolute*
number below is conditional on it. Every *comparative* number is not: both arms of every comparison
sit on the identical holdout.

---

## 1. Headline: the scarcity curve

Does adding synthetic attack data help when real attack data is scarce? Each of 100 cells
(5 k-values x 20 seeds) draws *k* of the 13 real fit campaigns, **refits the generator on only those
k**, regenerates a fresh 1,000-campaign corpus, and trains two detectors — real-only and
real+synthetic — scoring both once on the untouched holdout.
`python -m scripts.run_day4 sweep 20` · 100/100 cells, 0 failures.

| k | real only | real + synthetic | lift | p | seeds augmentation won |
|---:|---|---|---|---|---:|
| 1 | 0.240 ± 0.107 | 0.017 ± 0.014 | **−0.223** | 0.0002 | 1 / 20 |
| 2 | 0.424 ± 0.117 | 0.038 ± 0.017 | **−0.387** | <0.0001 | 0 / 20 |
| 4 | 0.340 ± 0.131 | 0.072 ± 0.018 | **−0.267** | 0.0007 | 3 / 20 |
| 8 | 0.593 ± 0.111 | 0.117 ± 0.023 | **−0.477** | <0.0001 | 1 / 20 |
| 13 | 0.658 ± 0.120 | 0.173 ± 0.018 | **−0.485** | <0.0001 | 3 / 20 |

*mean ± 95% CI across 20 seeds; p from a paired t-test on the per-seed lift* ·
`docs/figures/scarcity_curve.png`

**Synthetic augmentation does not fail to help. It actively destroys the detector, at every k.**
It wins 8 of 100 cells. The verdict is unchanged from the pre-fix version of this report, but it now
rests on 20 seeds and a paired test rather than on 5 seeds and a gap between means.

### Two things the leak was hiding

**1. The real-only arm is wildly unstable, and the leak concealed it.** At k=13 the real-only score
ranges **0.087 to 0.872** across seeds — sd 0.274, against the 0.022 CI reported before the fix.
Nothing differs between those runs but the GBT seed. The old ±0.022 interval was not a measurement
of a stable detector; it was the leak suppressing variance. **Any single-seed number from this
pipeline is one draw from a very wide distribution**, and that includes every number in earlier
versions of this report.

**2. The scarcity curve is not monotonic.** Real-only goes 0.240 → 0.424 → **0.340** → 0.593 →
0.658. Adding real campaigns from 2 to 4 *lowers* the mean. The pre-fix curve rose smoothly at every
step (0.430 → 0.590 → 0.816 → 0.825 → 0.903) and looked exactly like a scarcity curve is supposed
to look. **That tidiness was the leak.** With it removed, the honest statement is that more real
campaigns help on average but not reliably, and the seed dominates the trend at this sample size.

The augmented arm, by contrast, is *stable* — sd 0.018–0.052 at every k. It is consistently and
predictably bad. That combination (stable and low, against unstable and higher) is what §3 explains:
drowned in synthetic positives, the detector abandons `edge_rarity` for a coarse attribute and
becomes reliably mediocre instead of erratically useful.

---

## 2. Novelty / memorization

Read jointly with the lift, per the plan's decision rule.

Measured on the fit-window graph (§7.4), 10,000 campaigns / 161,587 rows.
Reproduce: `python -m scripts.run_day4 validate`.

| metric | value |
|---|---|
| max-Jaccard vs the 13 fit campaigns — median, p95, p99 | **0.0** |
| synth campaigns with max-Jaccard exactly 0 | 9,976 / 10,000 |
| `pct_near_duplicate` (≥ 0.5) | **0.0000** (0 campaigns) |
| `edge_novelty_rate` | **0.9982** |
| distinct synth edges / fit edges / overlap | 69,293 / 299 / **122** |

Not a single campaign now clears the near-duplicate threshold. The generator explores a smaller
edge set than it did on the full graph (69,293 vs 100,256 distinct edges) — the fit graph simply has
fewer edges to walk — but the novelty conclusion is unchanged and slightly stronger.

**The generator is not replaying the training campaigns.** It explores overwhelmingly new territory
within the real graph. Combined with §1 this is the plan's third case: **no lift + high novelty ⇒
honest negative.** The synthetic data explores real edges that are *not the discriminative ones*.

Novelty was never the problem. Fidelity is.

---

## 3. Structural-only ablation — is the signal real structure or the NTLM shortcut?

k=13, 10 seeds, fit-window graph. `python -m scripts.run_day4 ablation 10`.

| arm | with attributes | structural only |
|---|---|---|
| real only | 0.601 ± 0.199 | **0.688 ± 0.141** |
| real + synthetic | 0.171 ± 0.021 | **0.193 ± 0.020** |

**Detection survives, and the attributes were never carrying it.** Dropping the
`auth_type`/`logon_type` one-hots now *improves* the real-only detector by 0.087 and tightens its
interval. On the leaked graph this looked like a small 0.05 cost; the sign has flipped. Both arms
are better without attributes, so the one-hots are a liability for this task, not a shortcut worth
having.

Feature importances say what each arm is actually using:

| model | top feature | mass | structural / attribute |
|---|---|---|---|
| real only | `edge_rarity` | **0.971** | 0.980 / 0.020 |
| real + synthetic | `auth_type_NTLM` | **0.924** | 0.077 / 0.923 |

This is the one headline that the leak did **not** distort: the real-only model rests almost
entirely on graph position, and the augmented model abandons it for a single coarse tooling
attribute.

### Diagnosis: generative, not structural or attributional

The plan requires naming which limit binds. It is **generative**, and both sides of the evidence agree:

- **Not structural** — real-only holds **0.688** on graph structure alone, its best score in this
  report.
- **Not attributional** — dropping attributes *helps both arms*, so the one-hots were never what the
  real-only detector was standing on.
- **Generative** — at k=13 the augmented arm's positive class is ~95% synthetic (roughly 16,000 synth
  events against 650 real). `auth_type` is 100% NTLM on synthetic *and* on real red-team traffic, but
  only 5% of benign. Drowned in synthetic positives, the class-balanced GBT abandons `edge_rarity`
  for that single coarse one-hot — a rule with catastrophic precision at 1:4,000. And even with the
  crutch removed, the structural-only augmented arm sits at **0.193 against a 0.688 ceiling**,
  because the synthetic *structural* features are themselves wrong (§6).

---

## 4. V3 — TSTR and the three baselines

All at k=13 on the real holdout, 10 seeds, fit-window graph.
`python -m scripts.run_day4 baselines 10`.

| arm | AUC-PR |
|---|---|
| **#1 real-fit ceiling** (real only) | **0.601 ± 0.199** |
| **#2 real + naive synthetic** | **0.255 ± 0.028** |
| real + tuned synthetic | 0.171 ± 0.021 |
| **#2 TSTR — naive synthetic only** | 0.113 ± 0.037 |
| TSTR — tuned synthetic only | 0.033 ± 0.013 |
| **#3 rarity heuristic** — `edge_rarity`, no training at all | **0.0082** |
| chance | 0.000248 |

Three results deserve to lead, and two of them reverse what the leaked version of this report said.

**The training-free baseline collapsed.** Ranking by `edge_rarity` alone scores **0.0082**, not
0.530 — 33x chance rather than 2,100x, and a small fraction of the trained detector instead of 59%
of it. The old number was the report's most-quoted finding ("most of what the GBT learns is
available without learning anything"). **It was almost entirely the leak.** §7.4 shows the
mechanism: on the full-corpus graph no evaluation row could be new, so rarity separated the classes
by construction. It does not survive contact with a graph that stops at training time.

**Removing the sophistication now helps — significantly.** The naive generator (β=0, no hub
attraction, no credential bonus, random users) beats the tuned one on **10 of 10 seeds** in the
augmented arm (0.255 vs 0.171, paired t=+6.25, **p=0.0001**) and 9 of 10 in TSTR (0.113 vs 0.033,
p=0.0031). The previous version of this section concluded the two were "statistically
indistinguishable" and that "removing the sophistication neither helps nor hurts". On the leak-free
graph that is wrong: **the weighting and credential layers actively make the synthetic data worse.**

That is consistent with the structural argument in §6 and `data_insights.md` §4 — β>0 steers the
generator toward mega-hub servers, which is precisely the direction the real red team did not go. It
does not mean β=0 repairs augmentation: naive synthetic still costs 0.346 against real-only
(p=0.0081, winning 2 of 10 seeds). It means the sophistication is a net negative on top of an
approach that is already wrong.

**TSTR remains a failure.** A detector whose positives are purely synthetic is near-useless on real
attacks (0.033 / 0.113) — better than the collapsed rarity heuristic, which says more about the
heuristic than about the synthetic data.

**The real-fit ceiling is far lower and far less stable than reported**: 0.601 ± 0.199 against the
leaked 0.903 ± 0.022. The interval is ten times wider. See §1 and §9 on why that width, not the
mean, is now the dominant fact about this detector.

---

## 5. Base-rate sweep (test-time), rarest first

All 51 positives retained at every point; only benign is down-sampled.

10 seeds, fit-window graph. `python -m scripts.run_day4 baserate 10`.

| target | achieved | positives | negatives | AUC-PR |
|---|---|---:|---:|---|
| **sampled floor** | **1:4,032** | 51 | 205,612 | **0.601 ± 0.199** |
| 0.001 | 1:999 | 51 | 50,949 | 0.763 ± 0.175 |
| 0.01 | 1:99 | 51 | 5,049 | 0.957 ± 0.046 |

The 1:99 number is the one that would look best in a paper and is the least honest — it is bought
by making the haystack 40× thinner. Note how much of the apparent competence was the base rate
*and* the leak together: 0.999 at 1:99 on the leaked graph becomes 0.957 here, while the sampled
floor falls from 0.903 to 0.601. The easier the test, the better the leak's numbers held up —
which is what a leak does.

**But 1:4,032 is not a natural floor either, and this sweep only moves in the easy direction.** The
205,612 negatives are a *draw from* the holdout window, not the window. That window spans
`[1847858, 2557047]` = 709,189 s, 14.2% of the 58-day timeline, and the graph carries 1,051,430,459
events — so the window holds on the order of **149 million** events. (Estimated by assuming events
spread evenly over time; not confirmed by a corpus scan.) Against 51 positives that is a true base
rate near **1:2,900,000** — roughly **725× rarer than anything measured here**.

The extrapolation is not small. **3.47% of holdout benign already ranks at or above the red team's
5th-percentile `edge_rarity`** — 7,136 rows at 205,612 negatives, but on the order of 5.2 million at
full window volume, against 51 positives. AUC-PR at deployment volume would be far below the 0.601
measured here. How far is **not estimated here**; the sweep needed to answer it down-samples
positives instead of negatives, and was not run.
(`python -m scripts.run_day4 leakcheck`, which reports this on both graphs — the leak flattered it
too, at 4.60%.)

**Every absolute number in this report is conditional on the 1:4,032 sampled base rate.** The
comparative results are not — both arms of every comparison sit on the identical holdout, so the
augmentation deficit (§1) and the ablation contrasts (§3) are unaffected **by the base rate**. They
were emphatically not unaffected by the graph leak (§7.4), which is a separate matter and the reason
this report was rewritten.

---

## 6. V1 / V2 — fidelity

### V1 (SPEC exit criteria) — 161,587 rows / 10,000 campaigns

| check | result |
|---|---|
| timestamps strictly increasing within campaign, all Δt > 0 | **PASS** — 0 violations, min Δt = 1 |
| no campaign beyond the 58-day window | **PASS** — 0 violations, latest 4,976,443 ≤ 5,011,200 |
| every `(src,dst)` is a real graph edge | **PASS** — 0 violations |
| no credential used before acquisition | **N/A by architecture** |

The fourth item is **not** a passing assertion and is not counted as one. The fan-out generator has
no hops: every credential is harvested at the foothold at t=0, so use-before-acquisition cannot
occur by construction. Reporting it as "passed" would be dishonest.

### V2 — distributional fidelity

| field | KS D | p | median synth / real | mean synth / real |
|---|---|---|---|---|
| breadth | 0.154 | 0.875 | 9 / 9 | 16.2 / 27.9 |
| inter-event Δt | 0.002 | 1.000 | 90 / 90 | 342.4 / 344.3 |
| creds per campaign | 0.352 | 0.060 | 2 / 4 | 3.03 / 11.38 |
| **dst in-degree (JS divergence)** | **0.290** | — | target **< 0.1** | **MISS** |

`docs/figures/v2_*.png`

**The p-values are the least informative thing in this table.** With 13 real campaigns against
10,000 synthetic, KS has almost no power; a non-rejection is not a fit. The effect sizes:

- **dst in-degree is the largest gap and the only outright target miss.** Synthetic per-event target
  in-degree p25/50/75 = **7 / 1,544 / 12,101**; real = **7 / 12 / 40**. The generator hits mega-hub
  servers; the real red team hit obscure hosts — a ~130x gap at the median. The p25 agrees exactly,
  so the generator does place *some* attacks on obscure hosts; it is the middle and upper mass it
  cannot keep off the hubs. *Note §4: fixing this by setting β=0 does not repair augmentation — the
  gap is real but is not the whole story.*
- **creds per campaign p=0.054 is a non-rejection, not a fix.** Mean 2.98 vs 11.38, p95 9 vs 41. The
  Day-3 gap-#2 fix landed directionally but under-delivers ~4×.
- **breadth agrees in the body, not the tail.** Medians match exactly (9/9), but the means diverge
  (16.2 vs 27.9) — `min(k, len(targets))` clips the largest fan-outs.
- **inter-event Δt is near-tautological.** The generator bootstraps Δt by resampling the fit values,
  so D=0.004 confirms no placement bug; it is not independent evidence of temporal fidelity.
- **JS ≈ 0 on the constant categoricals proves nothing** and is deliberately not computed. Both
  sides are constant `(NTLM, Network, LogOn, Success)` by measurement; agreement is tautological.

---

## 7. Four defects found and fixed during this evaluation

### 7.1 The Day-3 benign sampler made the classes edge-disjoint

`src/benign.py` dropped every sampled benign row whose `(src,dst)` was a red-team edge, using the
edge set of all 701 events. The 51 holdout positives and 204,228 holdout negatives therefore shared
**exactly zero edges**. The feature-space region occupied by attack edges was guaranteed empty of
negatives in *both* the train and the eval split, so a detector learned "that region is positive" on
fit and was still right on holdout — not by generalizing, but because one global filter carved the
same hole twice.

**Measured cost of the artifact: GBT AUC-PR 0.999 → 0.894** once repaired. (Both figures, and the
0.550 below, were measured before §7.4 — on the leaked graph. They are recorded as the history of
how this defect was found and fixed, not as current scores.) `edge_rarity` alone ranked
at 0.550 throughout, which is what exposed it — a 0.999 trained score above a 0.550 single-feature
score was not credible.

`scripts/recover_hard_negatives.py` scans `auth.txt.gz` once and recovers the **3,662** benign events
on those 302 edges (2,278 fit / 1,384 holdout), anti-joining away the 701 real red-team events.
Holdout edge overlap went 0/39 → 7/39. **Every Day-4 number in this report uses the repaired splits.**

For 32 of 39 holdout attack edges there genuinely was no other traffic in the window — those are
rare edges where the only activity was the red team. That residual separability is a real property
of the data, not a construction artifact.

### 7.2 Synthetic generation was not reproducible across processes

`harvest_credentials` did `list(set-of-strings)` and indexed that order with the RNG. Python
randomizes string hashing per process, so a fixed seed produced a **different corpus every run** —
different credentials harvested → different `credential_bonus` in `target_weights` → different
targets and campaign sizes.

Caught by reconciling three runs that should have been the identical k=13 experiment: the real-only
arm matched bit-for-bit (it involves no generation), while `n_synth_pos` at seed 42 came out 14,071
vs 13,953 vs 14,804. Fixed with `sorted()`; the regression guard runs the generator in two
subprocesses under different `PYTHONHASHSEED` values, because an in-process check cannot catch this.

**All numbers in this report were produced before that fix.** They are therefore *independent draws*
rather than one reproducible corpus. This does not change any conclusion — it strengthens §1, since
all 15 independent k=13 draws show large harm — but it is why the augmented arm is quoted as an
interval. Re-running under the fix would tighten the intervals, not move them.

### 7.3 The base rate was described as a floor it is not

Covered in full in §5. The holdout's 1:4,032 ratio was reported as the deployment's "natural floor".
It is a **sampling choice**: the 205,612 negatives are a draw from a window holding ~149M events, so
the deployment rate is nearer **1:2.9M**, ~725x rarer. Every absolute number here is conditional on
the sampled rate; comparative numbers are not, because both arms of every comparison sit on the
identical holdout.

### 7.4 The graph was built from the whole corpus, including the future

The largest of the four, and the last one found. It invalidated every headline number in the
original version of this report.

`compute_edge_counts` and `compute_user_host_counts` scanned all 58 days. Those aggregates feed
`features.build_features`, which scores **holdout** rows — so an edge's weight, and therefore its
`edge_rarity`, was computed partly from the evaluation window itself. They also feed the walker, so
synthetic campaigns could route over edges that did not exist yet at fit time.

**The mechanism, measured** (`python -m scripts.run_day4 leakcheck`):

| rows sitting on an edge the graph has never seen | full-corpus graph | fit-window graph |
|---|---:|---:|
| holdout benign (205,612 rows) | **0.0000%** | 1.5082% (3,101) |
| holdout red team (51 rows) | **0.0000%** | 17.6471% (9) |

Read the first column. On the full-corpus graph **nothing in the evaluation set can be new**, for
either class — the graph contains every holdout row's own edge by construction. It was a perfect
oracle of which machine pairs would ever occur. `edge_rarity`'s extreme value means "this pair is
new", and that value was simply unreachable at evaluation time.

Bound the graph to the fit window and first-time-seen pairs reappear — on both sides. Those 3,101
benign rows now tie 9 of the 51 positives at maximum rarity. That is the real deployment problem,
and it is the one `data_insights.md` predicts: new hires, new projects and someone covering a
colleague produce first-time pairs constantly, and rarity alone cannot tell them from lateral
movement.

**What it cost.** Real-only AUC-PR at k=13 falls **0.9026 → 0.4122**. The training-free rarity
baseline falls **0.530 → 0.0082** — that number was the report's headline insight and was almost
entirely the leak.

**Why it survived.** A `t_hi` bound existed on two of the aggregates and was covered by tests. No
caller passed it. A guard nothing calls is documentation, not a fix. `run_day1.py` now builds a
`fit` arm from the same code path as the `full` arm, and every model-facing consumer reads only the
`fit` arm.

---

## 8. Insight for the README

> The synthetic data faithfully reproduces the actor's business-hours tradecraft, but timing is
> **deliberately excluded** from the detector — **a naive "flag out-of-hours activity" rule would
> miss this red team entirely.**

Behavioral fidelity and detection utility are separate axes. A feature can be behaviorally real and
detection-useless. Hour-of-day is exactly that here: benign traffic peaks in the same hours, and
there are zero off-hours attack examples to learn from.

**Correction to the stated Caveat B premise.** The carried-forward note said the real red team is
"~100% business-hours". Business hours here means **06:00–17:59** — a definition that used to live
only in an uncommitted script and is now pinned as `BUSINESS_HOURS` in `run_day4.py`, computed by
the `validate` task. Measured on `redteam_fit` it is **0.748** — 164 of 650 events fall in hours
18–23 (21:00 n=48, 20:00 n=38). Synthetic is 0.545 against a real *corpus* rate of 0.569: the
generator faithfully reproduces the **corpus** hourly profile by construction (`sample_start_time`
samples hour ∝ corpus volume) and therefore under-represents the red team's business-hour skew.
The drift is real; its magnitude is **0.55 vs 0.75**, not 0.55 vs ~1.0. `docs/figures/caveat_b_hourly.png`

This section is unaffected by the §7.4 fix: bounding the aggregates to the fit window moves the
corpus rate only 0.570 → 0.569, because the diurnal rhythm is stable across the 58 days.

Also carried forward: fan-out ≠ multi-hop · don't dress inference as fact · within-operator transfer
ceiling · attribute constancy is tool-specific.

---

## 9. Limitations

Stated first-class, not as an afterthought.

- **51 holdout positives.** Every interval here is wide. This is the binding statistical constraint
  and it cannot grow. The scarcity curve now uses 20 seeds precisely because removing the leak
  exposed how wide the per-seed spread really is (§1); the secondary tasks use 10.
- **Seed-to-seed variance is the dominant source of uncertainty, not the mean.** On the leak-free
  graph the real-only arm at k=13 swings across most of the available range depending only on which
  GBT seed is drawn. Any single-seed number from this pipeline, including in earlier versions of
  this report, should be read as one draw from a wide distribution.
- **Per-event altitude has a known ceiling.** Classifying single auth rows discards campaign
  structure, which is where lateral movement actually lives.
- **Within-operator transfer only.** Same red team, same toolkit, one engagement. A different
  attacker is **not** demonstrated. The 100%-NTLM constancy is tool-specific, not a property of
  lateral movement.
- **Credential→target assignment is a benign co-occurrence proxy** (open gap #1), not observed
  attacker credential use.
- **Single-foothold fan-out vs multi-origin reality** (open gap #5, deferred).
- **Benign label noise.** `redteam.txt` is detection-incomplete: "benign" may hide undetected
  attacks. `edge_rarity` is computed on a graph that deliberately still contains the red-team
  events — that is the realistic choice (a real deployment's baseline also contains the attack it
  has not yet detected), but it means the feature is measured against a contaminated baseline.
- **Negative-class sampling is now non-uniform.** Red-team edges have exhaustive benign coverage
  (§7.1) while all other edges are sampled at ~204k. This is conservative for detection claims —
  it over-represents benign traffic on attack edges — but it is not a uniform sample.
- **The 1:4,032 base rate is a sampling choice, not a floor** (§5). The holdout window holds ~149M
  events against 205,612 sampled negatives, implying a true rate near 1:2.9M — ~725× rarer. Absolute
  AUC-PR figures are conditional on the sampled rate; comparative results are not. The sweep in the
  *hard* direction was not run.
- **Broad foothold pool is deliberate** (open gap #3), not an oversight.
- **`credential_novelty` was removed for a reason that no longer holds.** It measured 0.0000 on all
  650 `redteam_fit` and 200k `benign_fit` rows but 0.3664 on synthetic, because
  `aggregates["user_host"]` was built over the whole corpus — every real row's pair was inside the
  map that was supposed to judge it novel, so the feature was definitionally dead on real data and a
  "this row is synthetic" marker in training. **Under the §7.4 fit-window bound that is no longer
  true**, and it becomes a genuine candidate feature. It is deliberately left out of this version so
  the leak correction is not confounded with a feature change; measure it separately.
- **Everything a model sees now comes from the fit window** (§7.4) — graph, credential-location map,
  hourly volume and attribute marginals. The full-corpus artifacts are retained for corpus
  description and for reproducing the leak contrast, and are read by nothing else.

---

## 10. Verdict

**Do not use this synthetic corpus to augment scarce real attack data.** It is worse than nothing:
it costs **0.22–0.48 AUC-PR** at every scarcity level tested (paired p < 0.001 at every k, winning
8 of 100 cells), and TSTR is near-useless.

The failure is **generative**. The corpus is structurally novel — 99.8% new real edges, not one
campaign above the near-duplicate threshold — but its per-event feature distribution does not match
the real attack, and at a ~20:1 synthetic-to-real ratio it dominates the positive class and pulls
the detector off `edge_rarity` (0.971 importance) and onto `auth_type_NTLM` (0.924).

**The hub-weighting layer is not neutral — it is actively harmful.** The naive generator (β=0, no
credential bonus) beats the tuned one on 10 of 10 seeds, p=0.0001. The earlier version of this
report concluded the two were indistinguishable; that was the leak. Turning the sophistication off
still does not rescue augmentation (naive costs 0.346 against real-only, p=0.008), so the right
reading is: **the weighting makes a wrong approach worse, and removing it does not make it right.**

The honest framing for anything built on this: **the generator reproduces THIS operator's campaign
shape, not its detectable structure.** It reproduces *permission* and *popularity*; the signal lives
in *habit*.

**What replaced the old closing claim.** The previous verdict ended by pointing at a training-free
rarity heuristic scoring 0.530 — "59% of the trained ceiling; before adding synthesis, that is the
bar." **That number was the leak.** On a graph that stops at training time, ranking by `edge_rarity`
alone scores **0.0082**. There is no cheap baseline that already solves this. The bar is lower than
v1 claimed, which makes the problem more open, not less — and it makes the wide, unstable real-only
arm (§1) the honest description of where detection currently stands.

---

*Reproduce (from the repo root, as modules — see README):*
*`python -m scripts.run_day4 sweep 20` (scarcity curve) ·*
*`python -m scripts.run_day4 all 10` (baselines, ablation, base-rate, validate) ·*
*`python -m scripts.run_day4 leakcheck` (§7.4) · `python -m pytest tests/ -q` (104 tests)*
