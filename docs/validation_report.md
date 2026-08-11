# Day 4 — Validation Report

**Synthetic lateral-movement data: does it substitute for scarce real attack data?**

Branch `day4-augmentation-eval` · 98 tests · evaluated 2026-08-11

---

## What this measures, and what it does not

The detector in this report is a **measuring instrument for synthetic-data quality**, not a
detection-science contribution. Nothing here claims a new way to find lateral movement. Every
number exists to answer one question: *is this synthetic data useful?*

The answer is **no**, and the evidence is unusually clean.

All numbers are AUC-PR (average precision) measured on the **real holdout**, never on synthetic
data. The holdout is **51 real red-team events / 6 campaigns against 205,612 benign events** — a
base rate of **0.000248 (1:4,033)**. Chance is 0.000248.

---

## 1. Headline: the scarcity curve

Does adding synthetic attack data help when real attack data is scarce? Each of 25 cells (5
k-values × 5 seeds) draws *k* of the 13 real fit campaigns, **refits the generator on only those
k**, regenerates a fresh 1,000-campaign corpus, and trains two detectors — real-only and
real+synthetic — scoring both once on the untouched holdout.

| k (real campaigns) | real only | real + synthetic | lift |
|---:|---|---|---|
| 1 | 0.430 ± 0.195 | 0.036 ± 0.050 | **−0.394** |
| 2 | 0.590 ± 0.160 | 0.152 ± 0.130 | **−0.439** |
| 4 | 0.816 ± 0.071 | 0.212 ± 0.097 | **−0.604** |
| 8 | 0.825 ± 0.070 | 0.221 ± 0.084 | **−0.604** |
| 13 | 0.903 ± 0.022 | 0.383 ± 0.073 | **−0.519** |

*mean ± 95% CI across 5 seeds (normal approximation; n=5 is small)* · `docs/figures/scarcity_curve.png`

**Synthetic augmentation does not fail to help. It actively destroys the detector, at every k.**

The real-only arm behaves exactly as a scarcity curve should — 0.43 at one campaign rising to 0.90
at thirteen. The augmented arm never comes close, and the damage does not shrink as real data grows.

Across **15 independent synthetic draws at k=13** (see §7 on why they are independent), the
augmented arm ranges **0.240 – 0.469** (mean 0.337, sd 0.067) against a real-only range of
**0.878 – 0.942**. The two distributions do not overlap at any point. This is not a marginal effect.

---

## 2. Novelty / memorization

Read jointly with the lift, per the plan's decision rule.

| metric | value |
|---|---|
| max-Jaccard vs the 13 fit campaigns — median, p95, p99 | **0.0** |
| synth campaigns with max-Jaccard exactly 0 | 9,988 / 10,000 |
| `pct_near_duplicate` (≥ 0.5) | **0.0002** (2 campaigns) |
| `edge_novelty_rate` | **0.9983** |
| distinct synth edges / fit edges / overlap | 100,256 / 299 / **166** |

The two "near-duplicates" are both breadth-1 campaigns whose single edge coincides with one of the
three singleton fit campaigns — coincidence at the smallest possible campaign size, not memorization.

**The generator is not replaying the training campaigns.** It explores overwhelmingly new territory
within the real graph. Combined with §1 this is the plan's third case: **no lift + high novelty ⇒
honest negative.** The synthetic data explores real edges that are *not the discriminative ones*.

Novelty was never the problem. Fidelity is.

---

## 3. Structural-only ablation — is the signal real structure or the NTLM shortcut?

| arm | with attributes | structural only |
|---|---|---|
| real only | 0.903 ± 0.022 | **0.853 ± 0.024** |
| real + synthetic | 0.290 ± 0.031 | **0.347 ± 0.026** |

**Detection survives.** Dropping the `auth_type`/`logon_type` one-hots costs the real-only detector
only 0.05. The real signal is graph structure, not the tooling artifact.

The augmented arm gets *better* when attributes are removed — it was leaning on a crutch the
real-only model never needed. Feature importances show why:

| model | top feature | mass | structural / attribute |
|---|---|---|---|
| real only | `edge_rarity` | **0.977** | 0.985 / 0.015 |
| real + synthetic | `auth_type_NTLM` | **0.944** | 0.055 / 0.945 |

### Diagnosis: generative, not structural or attributional

The plan requires naming which limit binds. It is **generative**, and both sides of the evidence agree:

- **Not structural** — real-only holds 0.853 on graph structure alone.
- **Not attributional** — dropping attributes costs real-only 0.05 and *helps* the augmented arm.
- **Generative** — at k=13 the augmented arm's positive class is **95.6% synthetic** (~14,286 synth
  events against 650 real). `auth_type` is 100% NTLM on synthetic *and* on real red-team traffic, but
  only 5% of benign. Drowned in synthetic positives, the class-balanced GBT abandons `edge_rarity`
  for that single coarse one-hot — a rule with catastrophic precision at 1:4,000. And even with the
  crutch removed, the structural-only augmented arm sits at 0.347 against a 0.853 ceiling, because
  the synthetic *structural* features are themselves wrong (§6).

---

## 4. V3 — TSTR and the three baselines

All at k=13, on the real holdout.

| arm | AUC-PR |
|---|---|
| **#1 real-fit ceiling** (real only) | **0.903 ± 0.022** |
| **#3 rarity heuristic** — `edge_rarity`, no training at all | **0.530** |
| real + tuned synthetic | 0.339 ± 0.039 |
| **#2 real + naive synthetic** | 0.343 ± 0.061 |
| TSTR — tuned synthetic only | 0.060 ± 0.069 |
| **#2 TSTR — naive synthetic only** | 0.118 ± 0.096 |
| chance | 0.000248 |

Two results deserve to lead.

**A training-free ranking by edge rarity alone reaches 0.530 — 59% of the trained detector, and
~2,100× chance.** Most of what the GBT "learns" is available without learning anything. Any claim
that this task needs machine learning has to clear that bar first.

**TSTR is a failure.** A detector whose positives are purely synthetic is near-useless on real
attacks (0.060 / 0.118), with confidence intervals that touch zero.

**Baseline #2 refutes the obvious hypothesis.** We expected the `beta=1` in-degree weighting — which
drives the generator onto mega-hub servers — to be the culprit. It is not. The naive generator
(β=0, no hub attraction, no credential restriction, random users) is statistically indistinguishable
from the tuned one in both the augmented arm (0.343 vs 0.339) and TSTR (0.118 vs 0.060, CIs
overlapping). **Removing the sophistication neither helps nor hurts.** Whatever breaks this data is
not the weighting layer.

---

## 5. Base-rate sweep (test-time), rarest first

All 51 positives retained at every point; only benign is down-sampled.

| target | achieved | positives | negatives | AUC-PR |
|---|---|---:|---:|---|
| **natural floor** | **1:4,033** | 51 | 205,612 | **0.903 ± 0.022** |
| 0.001 | 1:999 | 51 | 50,949 | 0.974 ± 0.011 |
| 0.01 | 1:99 | 51 | 5,049 | 0.999 ± 0.002 |

The 1:99 number is the one that would look best in a paper and is the least honest — it is bought
by making the haystack 40× thinner. **The natural floor is the only number that describes the
deployment this data came from.**

---

## 6. V1 / V2 — fidelity

### V1 (SPEC exit criteria) — 144,228 rows / 10,000 campaigns

| check | result |
|---|---|
| timestamps strictly increasing within campaign, all Δt > 0 | **PASS** — 0 violations, min Δt = 1 |
| no campaign beyond the 58-day window | **PASS** — 0 violations, latest 4,955,790 ≤ 5,011,200 |
| every `(src,dst)` is a real graph edge | **PASS** — 0 violations |
| no credential used before acquisition | **N/A by architecture** |

The fourth item is **not** a passing assertion and is not counted as one. The fan-out generator has
no hops: every credential is harvested at the foothold at t=0, so use-before-acquisition cannot
occur by construction. Reporting it as "passed" would be dishonest.

### V2 — distributional fidelity

| field | KS D | p | median synth / real | mean synth / real |
|---|---|---|---|---|
| breadth | 0.188 | 0.684 | 9 / 9 | 14.4 / 27.9 |
| inter-event Δt | 0.004 | 1.000 | 87 / 90 | 340.1 / 344.3 |
| creds per campaign | 0.358 | 0.054 | 2 / 4 | 2.98 / 11.38 |
| **dst in-degree (JS divergence)** | **0.350** | — | target **< 0.1** | **MISS** |

`docs/figures/v2_*.png`

**The p-values are the least informative thing in this table.** With 13 real campaigns against
10,000 synthetic, KS has almost no power; a non-rejection is not a fit. The effect sizes:

- **dst in-degree is the largest gap and the only outright target miss.** Synthetic per-event target
  in-degree p25/50/75 = **143 / 11,659 / 14,221**; real = **10 / 20 / 104**. The generator hits
  mega-hub servers; the real red team hit obscure hosts. Robust to binning (0.318 @10 bins → 0.391
  @50). *Note §4: fixing this by setting β=0 does not repair augmentation — the gap is real but is
  not the whole story.*
- **creds per campaign p=0.054 is a non-rejection, not a fix.** Mean 2.98 vs 11.38, p95 9 vs 41. The
  Day-3 gap-#2 fix landed directionally but under-delivers ~4×.
- **breadth agrees in the body, not the tail.** Medians match exactly (9/9), p25/p75 match, but p95
  is 39 synth vs 120 real — `min(k, len(targets))` clips the largest fan-outs.
- **inter-event Δt is near-tautological.** The generator bootstraps Δt by resampling the fit values,
  so D=0.004 confirms no placement bug; it is not independent evidence of temporal fidelity.
- **JS ≈ 0 on the constant categoricals proves nothing** and is deliberately not computed. Both
  sides are constant `(NTLM, Network, LogOn, Success)` by measurement; agreement is tautological.

---

## 7. Two defects found and fixed during this evaluation

### 7.1 The Day-3 benign sampler made the classes edge-disjoint

`src/benign.py` dropped every sampled benign row whose `(src,dst)` was a red-team edge, using the
edge set of all 701 events. The 51 holdout positives and 204,228 holdout negatives therefore shared
**exactly zero edges**. The feature-space region occupied by attack edges was guaranteed empty of
negatives in *both* the train and the eval split, so a detector learned "that region is positive" on
fit and was still right on holdout — not by generalizing, but because one global filter carved the
same hole twice.

**Measured cost of the artifact: GBT AUC-PR 0.999 → 0.894** once repaired. `edge_rarity` alone ranked
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

---

## 8. Insight for the README

> The synthetic data faithfully reproduces the actor's business-hours tradecraft, but timing is
> **deliberately excluded** from the detector — **a naive "flag out-of-hours activity" rule would
> miss this red team entirely.**

Behavioral fidelity and detection utility are separate axes. A feature can be behaviorally real and
detection-useless. Hour-of-day is exactly that here: benign traffic peaks in the same hours, and
there are zero off-hours attack examples to learn from.

**Correction to the stated Caveat B premise.** The carried-forward note said the real red team is
"~100% business-hours". Measured on `redteam_fit` it is **0.748** — 164 of 650 events fall in hours
18–23 (21:00 n=48, 20:00 n=38). Synthetic is 0.549 against a real *corpus* rate of 0.570: the
generator faithfully reproduces the **corpus** hourly profile by construction (`sample_start_time`
samples hour ∝ corpus volume) and therefore under-represents the red team's business-hour skew.
The drift is real; its magnitude is **0.55 vs 0.75**, not 0.55 vs ~1.0. `docs/figures/caveat_b_hourly.png`

Also carried forward: fan-out ≠ multi-hop · don't dress inference as fact · within-operator transfer
ceiling · attribute constancy is tool-specific.

---

## 9. Limitations

Stated first-class, not as an afterthought.

- **51 holdout positives.** Every interval here is wide and every CI uses n=5 seeds with a normal
  approximation. This is the binding statistical constraint and it cannot grow.
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
- **Holdout base-rate floor ~1:4,000.** Real deployments are rarer still.
- **Broad foothold pool is deliberate** (open gap #3), not an oversight.
- **`credential_novelty` was specified by the plan and removed.** It measured 0.0000 on all 650
  `redteam_fit` and 200k `benign_fit` rows but 0.3664 on synthetic, because `aggregates["user_host"]`
  is built from a corpus that contains the red-team events — making it definitionally dead on real
  data and a "this row is synthetic" marker in training. Rationale is recorded in
  `src/features.py`'s docstring with a regression test.
- **Numbers predate the determinism fix** (§7.2).

---

## 10. Verdict

**Do not use this synthetic corpus to augment scarce real attack data.** It is worse than nothing:
it costs 0.39–0.60 AUC-PR at every scarcity level tested, and TSTR is near-useless.

The failure is **generative**, and it is not the hub-weighting layer — turning that off changes
nothing. The corpus is structurally novel (98% new real edges, no memorization) but its per-event
feature distribution does not match the real attack, and at a 20:1 synthetic-to-real ratio it
dominates the positive class and pulls the detector onto a coarse tooling attribute.

The honest framing for anything built on this: **the generator reproduces THIS operator's campaign
shape, not its detectable structure.**

The most useful number in this report is the one that required no machine learning at all — **a
rarity heuristic at 0.530, 59% of the trained ceiling.** Before adding synthesis, that is the bar.

---

*Reproduce: `python scripts/run_day4.py` (scarcity sweep) · `python scripts/run_day4.py all`
(baselines, ablation, base-rate) · `python -m pytest tests/ -q` (98 tests)*
