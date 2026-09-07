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
