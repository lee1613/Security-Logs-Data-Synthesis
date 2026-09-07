# V1 — how this generator was built, and what it is worth

A process record for the first complete version of the synthetic lateral-movement pipeline.
It exists to be argued with: v2 should start from the specific place v1 fails, not from scratch.

Companion documents — this one does not repeat them:

| Document | Covers |
|---|---|
| [`SPEC_synthetic_lateral_movement.md`](../SPEC_synthetic_lateral_movement.md) | What was specified up front |
| [`data_insights.md`](data_insights.md) | What the LANL corpus actually looks like |
| [`validation_report.md`](validation_report.md) | The measurements and their caveats |

---

## 1. What v1 is

Four stages, each a script, each writing artifacts the next one reads.

| Stage | Script | Produces |
|---|---|---|
| 1 | `run_day1.py` | Corpus aggregates, machine graph, red-team recovery, frozen fit/holdout split, behavioral profile |
| 2 | `run_day2.py` | Fan-out campaign corpus from the profile |
| 3 | `run_day3.py` | Campaigns placed on the timeline, emitted as labeled auth rows at three base rates |
| 4 | `run_day4.py` | Features, detector, scarcity curve, baselines, ablation, base-rate sweep |

The generator itself is a **weighted fan-out walk on real graph edges**. It fits six distributions
from the real red-team fit campaigns — chain length, inter-hop delay, seed-host degree, target
in-degree, conditional row attributes, credential reuse — then walks the machine graph sampling
targets from a foothold's *real* out-edges, weighted by in-degree (`beta`) and by whether a harvested
credential has been seen on the target (`credential_bonus`).

Two constraints were deliberate from the start:

- **Only real edges.** A synthetic campaign never invents a machine pair that does not exist in the
  corpus. This is what keeps the output plausible at the row level.
- **The synthetic budget is constant across every scarcity point.** 1,000 campaigns whether the
  generator was fit on 1 real campaign or 13. If the budget tracked *k*, the scarcity curve would
  confound "less real data" with "less synthetic data" and stop isolating *k*.

---

## 2. How it is measured

The detector is a **measuring instrument, not a contribution.** Nothing here claims a new way to find
lateral movement. Every number answers one question: *is this synthetic data useful?*

The design that makes that question answerable:

- **Every score is AUC-PR on the real holdout.** Synthetic data is never scored, only trained on.
- **The split is frozen and the files are read-only on disk.** The earliest 70% of red-team campaigns
  are fit; the rest are holdout. `run_day1.py` chmods them read-only so no later stage can drift.
- **The scarcity curve refits the generator per cell.** Drawing *k* of 13 real campaigns and then
  generating from a generator fit on all 13 would leak the other campaigns into the synthetic arm.
  Each cell refits on only its own *k*.
- **Three baselines, because "it helps" needs something to beat.** A real-only arm (ceiling), a
  deliberately naive generator (does the sophistication earn its keep?), and a training-free ranking
  by `edge_rarity` alone (does this need machine learning at all?).
- **TSTR** — train on synthetic, test on real — as the hardest version of the question.

### The honest-outcome policy

Nothing was tuned. No hyperparameter search, no threshold picking, no re-running until the curve
improved. Where a check failed it is reported as failed. This mattered more than expected: v1's
result is negative, and the temptation to keep adjusting until it was not is exactly what the policy
is for.

---

## 3. Defects found, in the order they were found

Every one of these made the numbers look *better*. That is the pattern worth internalising — a
pipeline's bugs are not randomly signed, because a result that looks good gets checked less.

### 3.1 The benign sampler made the classes edge-disjoint

Day 3's sampler excluded every benign row sitting on a known red-team edge. Positives and negatives
then shared *zero* machine pairs, so the detector could separate them by finding a hole in feature
space rather than by finding the attack. Worth a spurious **0.999 → 0.894**.
Fixed by `scripts/recover_hard_negatives.py`, which puts those rows back as hard negatives.

### 3.2 Generation was not reproducible across processes

Credential harvesting indexed into a `set` of strings. Python randomizes string hashing per process,
so a fixed seed produced a different corpus on every run — and any cross-run comparison was
meaningless. Fixed with `sorted()`, guarded by a regression test that runs generation in a
subprocess.

### 3.3 The base rate was described as a floor it is not

The holdout's 1:4,033 base rate was reported as the "natural floor" — the rate the deployment would
actually see. It is not. The 205,612 negatives are a *draw from* the holdout window, which holds on
the order of 149 million events. The true rate is nearer **1:2,900,000**, ~725× rarer. Every absolute
number is conditional on the sampled rate; comparative numbers are not, since both arms of every
comparison sit on the identical holdout.

### 3.4 The graph was built from the whole corpus, including the future

The one that mattered most, and the last one found.

`compute_edge_counts` and `compute_user_host_counts` scanned all 58 days. Those aggregates feed
`features.build_features`, which scores **holdout** rows. So an edge's weight — and therefore its
`edge_rarity`, the single strongest feature — was computed partly from the evaluation window itself.
The detector was being asked how rare a machine pair is, and answering with a count that already
included the future.

It also reached the generator: the walker traverses the same graph, so synthetic campaigns could
route over edges that did not exist yet at fit time.

**Measured cost: real-only AUC-PR 0.9026 → 0.4122.** Slightly over half the headline number was the
leak.

The fix is a `t_hi` bound on every aggregate, and — the part that was actually missing — passing it.
The guard existed for two functions but no caller used it, so the shipped pipeline still built and
consumed the leaked graph. `run_day1.py` now builds both arms from one code path: a `full` arm for
corpus description and for reproducing this contrast, and a `fit` arm that is the only thing any
model is allowed to see.

### 3.5 Two report sections had no code behind them

`src/validate.py` computes the novelty and V1/V2 fidelity statistics, and nothing in the repository
called it. Sections 2 and 6 of the validation report were produced by a script that was never
committed. The numbers were not wrong, but they were not reproducible either, which for a report
whose entire claim is methodological is close to the same thing. `run_day4.py validate` is now that
caller.

The general lesson, and the reason this is listed with the defects: **an uncommitted analysis script
is an unreproducible result wearing a table.** If a number reaches a report, the thing that computed
it belongs in the repo.

---

## 4. What v1 is actually worth

Everything below is measured on the leak-free fit-window graph, 20 seeds for the scarcity curve and
10 for the rest. The "before" column is the same pipeline on the full-corpus graph — i.e. what this
project believed a week ago.

### The verdict held. Almost nothing supporting it did.

| | before (leaked) | after (fit window) |
|---|---|---|
| real-only ceiling, k=13 | 0.903 ± 0.022 | **0.658 ± 0.120** |
| real + synthetic, k=13 | 0.383 ± 0.073 | 0.173 ± 0.018 |
| lift at k=13 | −0.519 | **−0.485** (p<0.0001, 20 seeds) |
| rarity heuristic, no training | 0.530 | **0.0082** |
| naive vs tuned generator | indistinguishable | **naive wins 10/10** (p=0.0001) |
| dropping attribute one-hots | costs 0.05 | **gains 0.087** |
| real-only spread across seeds | sd 0.022 | **sd 0.274** (range 0.087–0.872) |

**The headline survived**: synthetic augmentation costs 0.22–0.48 AUC-PR at every k, p<0.001
throughout, winning 8 of 100 cells. It is now supported by 20 seeds and a paired test instead of 5
seeds and a gap between means.

**Three conclusions did not survive.**

1. **The training-free rarity baseline was the leak.** 0.530 → 0.0082. This had been the project's
   most-quoted result — "most of what the GBT learns is available without learning anything". It is
   gone. There is no cheap baseline that already solves this.
2. **The tuned generator is worse than a naive one.** Previously "removing the sophistication
   neither helps nor hurts"; now the naive generator wins 10 of 10 seeds. The weighting layers are a
   net negative.
3. **The scarcity curve is not a curve.** Real-only runs 0.240 → 0.424 → **0.340** → 0.593 → 0.658.
   Going from 2 real campaigns to 4 *lowers* the mean. The pre-fix version rose smoothly at every
   step and looked exactly like a textbook scarcity curve. That tidiness was the leak.

### The most important number is a standard deviation

At k=13 the real-only detector scores anywhere from **0.087 to 0.872** depending on nothing but the
GBT seed. Before the fix that interval was ±0.022 and the detector looked stable.

This reframes what the whole pipeline can support. It is a usable instrument for *comparing two arms
on identical data* — which is what every claim here actually is, and why the comparative results
hold up. It is not an instrument for stating what AUC-PR this detector achieves. Any single-seed
number it produces is one draw from a very wide distribution.

The augmented arm, by contrast, is stable at every k (sd 0.018–0.052). Consistently mediocre. That
asymmetry *is* the finding: synthetic positives don't add signal, they add ballast — the detector
stops betting on `edge_rarity` (importance 0.971) and settles on `auth_type_NTLM` (0.924), which is
reliable and nearly worthless at a 1:4,000 base rate.



---

## 5. Where v1 fails, stated precisely

This is the section to argue with. Everything above is bookkeeping; this is the input to v2.

### The defect is target selection, and it is upstream of every knob

The generator picks a foothold, then samples targets from that foothold's **real out-edges**,
weighted toward high in-degree. That reproduces two things faithfully — **permission** (the pair is
allowed, and exists in the corpus) and **popularity** (busy machines get picked). It reproduces
neither of the things that make a real intrusion detectable.

Per-event target in-degree, measured on the **fit graph** — the one every model now sees:

| | n | p25 | median | p75 |
|---|---:|---:|---:|---:|
| every machine (in-degree) | 15,093 | 1 | **4** | 5 |
| machine a **benign event** targets | 204,228 | 792 | **12,101** | 12,529 |
| machine the **red team** targeted (fit split) | 650 | 7 | **12** | 40 |
| machine the **generator** targeted | 161,587 | 7 | **1,544** | 12,101 |

*(`data/derived/day4_validate.json`; the equivalent full-corpus-graph table is
[`data_insights.md`](data_insights.md) §4, where the same gap reads 20 vs 11,659)*

The generator's median target is **~130x** more connected than the red team's. Note the p25 now
agrees exactly (7 vs 7): the generator does place *some* attacks on obscure hosts. It is the middle
and upper mass that is wrong — it cannot stay off the hubs.

The generator aims where legitimate traffic already flows. It manufactures **normal-shaped traffic
labelled "attack"** — which does not merely fail to teach the detector, it teaches it something
false: that hub-directed traffic, i.e. most legitimate traffic, is malicious.

**Turning off the hub weighting does not fix this.** Baseline #2 sets β=0 and removes the credential
bonus, and it is statistically indistinguishable from the tuned generator in both the augmented arm
and TSTR. β=0 yields *uniform over permitted*, and uniform-over-permitted is still not habit.

### The thing that has to be modelled instead

> **Permission is broad; behaviour is narrow.**

A domain account *could* reach hundreds of machines. In practice a person touches three or four,
stably, for weeks. Every red-team login in this corpus was **permitted** — real stolen credentials,
domain approved every one, all `Success`. Authentication worked exactly as designed. The only
separator is prior history: one pair had ~219,000 previous events, the other about 3.

So the target for v2 is not "generate plausible auth rows" — v1 already does that, and V1's hard
constraints pass. It is:

1. **Model habit, not permission.** A per-credential distribution over *its own* stable host set,
   not over the graph's popular nodes. The gap between what an account may do and what it does is
   the entire signal.
2. **Then make the attack deviate from habit** by a small count of structurally distinct events —
   sideways, workstation-to-workstation, across the grain of the vertical workstation→server flow.

A generator that samples a foothold's real out-edges cannot express (1), because the out-edge set is
the permission set. This is an architectural limit, not a parameter to tune.

### Three concrete leads

- **`credential_novelty` is now a real feature.** It was measured at exactly 0.0000 on every real row
  and dropped as a synthetic-row marker — but that was an artifact of building `user_host` over the
  whole corpus, so every real row's pair was inside the map judging it novel. Under the `t_hi` bound
  a holdout pair is no longer tautologically present. Measure it on its own; it is deliberately not
  in v1 so the leak correction stays unconfounded with a feature change.
- **Two strong signals are computed nowhere.** `is_self_auth` (0.00% red team / 53.48% benign) and
  `is_machine_account` (0.00% / 60.32%) would remove a large share of the haystack for almost no
  cost. Both carry the label-construction caveat in `data_insights.md` §2, which is exactly why they
  deserve a deliberate decision rather than silent omission.
- **Thirteen campaigns is a weak foundation for distribution fitting**, whatever is being fitted.
  Any v2 that keeps a fit-distributions architecture inherits this. Worth deciding up front whether
  the generator should be fitted at all, or specified from the structural argument above.

### The bar to clear

Not the trained detector — the **training-free one**. Ranking rows by `edge_rarity` alone, with no
model, gets a large fraction of the trained detector's score. Any synthesis work has to beat that
before it has earned its complexity, and the honest comparison is against the *rarity heuristic*,
not against chance.

---

## 6. What would have caught these problems sooner

Ordered by how much time each would have saved.

1. **Assert the provenance of every artifact a model consumes.** The leak survived because
   `graph.pkl` was correct-looking and silently wrong. A one-line check — the graph's total edge
   weight equals the event count inside the fit window — falsifies it immediately. That check is
   what confirmed the corrected artifact here, and it is cheap enough to run every time.
2. **Grep for callers before trusting a guard.** The `t_hi` parameter existed, was tested, and was
   passed by nobody. A guard nothing calls is documentation, not a fix.
3. **Never let a number reach a report from an uncommitted script.** Two whole sections rested on
   code that no longer existed.
4. **Report a win count next to every mean.** A mean lift of −0.24 and "augmentation lost on 5 of 5
   seeds" are different claims; only one of them was ever true, and the mean alone cannot tell you
   which.
