# Five evaluation defects, and what each one cost

A failure record from **v0** of the synthetic lateral-movement project (LANL auth corpus,
2026-08 → 2026-09). It is written to be read on its own, by someone who did not build this.

*Standalone extract of [`validation_report.md`](validation_report.md) §7 and
[`v0_process.md`](v0_process.md) §3, which remain the sources of truth for these numbers.*

The project asked one question: *if you manufacture fake attack logins and train a detector on
them, does the detector get better at catching real attacks?* The answer turned out to be no. But
before that answer could be trusted, five defects had to be found in the **measurement apparatus**
itself — not in the generator, in the ruler used to grade it.

> **The pattern worth taking away:** all five made the results look *better*. Not one made them look
> worse. That is not chance. A pipeline's bugs are not randomly signed, because a result that looks
> good gets checked less. Every hour of verification here was spent on numbers that had already
> passed the "does this seem plausible?" test — and four of the five were sitting inside numbers
> that had already been published in a report.

---

## At a glance

| # | Defect | What it inflated | Measured cost | Fixed in |
|---|---|---|---|---|
| 1 | Benign sampler made the two classes edge-disjoint | The detector's score | AUC-PR 0.999 → **0.894** | `c045b1f` |
| 2 | Generation not reproducible across processes | Confidence in every comparison | Same seed → 13,953 / 14,071 / 14,804 positives | `302b3b6` |
| 3 | A sampled base rate described as a deployment floor | Every absolute number's meaning | Claim was ~**725×** off | docs (§5) |
| 4 | Graph built from the whole corpus, including the future | Everything | AUC-PR 0.9026 → **0.4122**; rarity baseline 0.530 → **0.0082** | `78ec3f9`, `47d63a4` |
| 5 | Two report sections produced by uncommitted code | Reproducibility, not the values | No score change; two sections unverifiable | `8206639` |

Defect 4 is the one that matters most: it invalidated every headline number in the first version of
the validation report and reversed two of its conclusions outright.

---

## 1. The benign sampler made the two classes edge-disjoint

**In plain terms.** To train a detector you need examples of attacks (positives) and examples of
normal traffic (negatives). The code that collected normal traffic threw away every normal login
that happened between a pair of machines the attackers had also used. Sounds tidy. It meant attack
examples and normal examples shared *zero* machine pairs — so the detector never had to learn what
an attack looks like. It only had to learn which machine pairs were missing from the normal pile.

**Mechanically.** `src/benign.py` dropped every sampled benign row whose `(src,dst)` appeared in the
edge set of all 701 red-team events. The 51 holdout positives and 204,228 holdout negatives shared
exactly zero edges. One global filter carved the same hole in feature space *twice* — once in the
training split, once in the exam — so a detector that memorized "this region is positive" was still
right at test time, without generalizing at all.

**How it was caught.** A single-feature ranking (`edge_rarity`, no model at all) scored 0.550 while
the fully trained model scored 0.999. That gap is not credible. A trained model beating a
one-feature heuristic by that much means either the heuristic is being denied something, or the
model is seeing something it should not.

**Cost.** GBT AUC-PR **0.999 → 0.894** once repaired.

**Fix.** `scripts/recover_hard_negatives.py` rescans the raw corpus and puts back the **3,662**
benign events on those 302 edges (2,278 fit / 1,384 holdout), anti-joining away the 701 real
red-team rows. Holdout edge overlap went 0/39 → 7/39.

**Residual, stated honestly.** For 32 of 39 holdout attack edges there genuinely was no other
traffic in the window. That separability is a real property of the data, not an artifact.

---

## 2. Synthetic generation was not reproducible across processes

**In plain terms.** The generator took a random seed, so the same seed was supposed to produce the
same fake data every time. It did not. Two runs with identical settings produced different corpora,
which quietly means every before/after comparison was comparing two different things.

**Mechanically.** `harvest_credentials` did `list(set-of-strings)` and used the RNG to index into
that order. Python randomizes string hashing per process, so set iteration order changes between
runs — different credentials harvested → different `credential_bonus` in `target_weights` →
different targets and different campaign sizes.

**How it was caught.** By reconciling three runs that should have been the identical k=13
experiment. The real-only arm matched bit-for-bit (no generation involved), while the synthetic
positive count at seed 42 came out **14,071 vs 13,953 vs 14,804**. The arm that involved none of the
generator was the control that exposed it.

**Cost.** No single number, but a category of claim: results produced before the fix are
*independent draws*, not one reproducible corpus. This is why the augmented arm was quoted as an
interval rather than a point.

**Fix.** `sorted()`. The regression guard runs generation in **two subprocesses under different
`PYTHONHASHSEED` values** — an in-process test cannot catch this class of bug, because the hash seed
is fixed for the life of a process.

---

## 3. A sampled base rate described as a deployment floor

**In plain terms.** Attacks are rare, and exactly how rare changes how good any detector looks. The
report said the test set's ratio of 1 attack per 4,032 events was the "natural floor" — what a real
deployment would see. It was not. It was an artifact of how many normal events happened to be
sampled.

**Mechanically.** The 205,612 negatives are a *draw from* the holdout window, which holds on the
order of **149 million** events. The real deployment rate is nearer **1:2,900,000** — about **725×
rarer** than reported.

**Cost.** No score moved, but the meaning of every absolute number did. Absolute figures are
conditional on the sampled rate. Comparative figures are not, because both arms of every comparison
sit on the identical holdout — which is the only reason the project's conclusions survived this one.

**Fix.** Documentation, plus a base-rate sweep so the sensitivity is visible rather than asserted.

---

## 4. The graph was built from the whole corpus, including the future

The largest of the five, and the last one found.

**In plain terms.** The detector's strongest signal was "how unusual is it for these two machines to
talk?" That question was answered using a map of machine-to-machine traffic built from **all 58
days** — including the days held back as the exam. So when the detector asked "have I ever seen this
pair before?", the map answered using the exam itself. Nothing in the test set could possibly be
new.

**Mechanically.** `compute_edge_counts` and `compute_user_host_counts` scanned the full corpus.
Those aggregates feed `features.build_features`, which scores **holdout** rows — so an edge's weight
and therefore its `edge_rarity` was computed partly from the evaluation window. They also feed the
walker, so synthetic campaigns could route over edges that did not exist yet at fit time.

**The mechanism, measured** (`python -m scripts.run_day4 leakcheck`):

| rows sitting on an edge the graph has never seen | full-corpus graph | fit-window graph |
|---|---:|---:|
| holdout benign (205,612 rows) | **0.0000%** | 1.5082% (3,101) |
| holdout red team (51 rows) | **0.0000%** | 17.6471% (9) |

Read the first column. Zero, for both classes. The graph contained every holdout row's own edge by
construction — a perfect oracle of which machine pairs would ever occur. `edge_rarity`'s extreme
value means "this pair is new", and that value was simply unreachable at evaluation time.

Bound the graph to the training window and first-time pairs reappear on both sides. Those 3,101
benign rows now tie 9 of the 51 positives at maximum rarity — which is the actual deployment
problem: new hires, new projects, and someone covering a colleague produce first-time pairs
constantly, and rarity alone cannot separate them from an intruder.

**Cost.** Real-only AUC-PR at k=13 fell **0.9026 → 0.4122** — slightly over half the headline number
was the leak. The training-free rarity baseline fell **0.530 → 0.0082**; that number had been the
project's most-quoted result ("most of what the model learns is available without learning
anything") and was almost entirely contamination. Two further conclusions reversed: the tuned
generator turned out to be *worse* than a deliberately naive one (10 of 10 seeds, p=0.0001), and the
detector's seed-to-seed spread went from sd 0.022 to **sd 0.274** (range 0.087–0.872).

**Why it survived.** A `t_hi` time bound existed on two of the aggregate functions. It had unit
tests. It passed them. **No caller ever passed the argument.** The shipped pipeline built and
consumed the leaked graph anyway.

> **A guard that nothing calls is documentation, not a fix.**

**Fix.** `run_day1.py` now builds both arms from one code path: a `full` arm for corpus description
(and for reproducing this contrast), and a `fit` arm that is the only graph any model is permitted
to see. Every model-facing consumer was rerouted (`78ec3f9`, `47d63a4`), and a `leakcheck` task
turns the contamination into a standing measurement rather than a memory.

---

## 5. Two report sections had no code behind them

**In plain terms.** Two sections of the validation report contained tables of numbers that no
committed code could regenerate. They came from a script that was never checked in.

**Mechanically.** `src/validate.py` computes the novelty and V1/V2 fidelity statistics, and nothing
in the repository called it. The numbers were not wrong — but they were not reproducible either,
which for a report whose entire claim is methodological is nearly the same thing.

**Fix.** `run_day4.py validate` is now that caller (`8206639`).

> **An uncommitted analysis script is an unreproducible result wearing a table.** If a number reaches
> a report, the thing that computed it belongs in the repo.

---

## What would have caught these sooner

Ordered by time saved:

1. **Assert the provenance of every artifact a model consumes.** The leak survived because
   `graph.pkl` looked correct and was silently wrong. One line — *the graph's total edge weight
   equals the event count inside the training window* — falsifies it immediately. That check is what
   confirmed the corrected artifact, and it is cheap enough to run on every build.
2. **Grep the callers before trusting a guard.** Existence and test coverage are not use. Defect 4
   had both and still shipped.
3. **Distrust results you like.** All five defects inflated the numbers. Budget verification in
   proportion to how pleasing a result is, not how suspicious it looks.
4. **Report a spread and a win count next to every mean.** A mean lift of −0.24 and "augmentation
   lost on 5 of 5 seeds" are different claims; the mean alone cannot tell you which one you have.
   And a detector quoted at ±0.022 turned out to swing between 0.087 and 0.872 on seed alone.
5. **Keep a control arm that exercises less machinery than the arm under test.** Defect 1 was caught
   by a one-feature heuristic, defect 2 by the arm with no generator in it. Both times, the simpler
   thing disagreeing with the fancier thing was the alarm.

---

## Provenance of the numbers here

The 0.999 → 0.894 and 0.550 figures in defect 1 were measured *before* defect 4 was found — on the
leaked graph. They are recorded as the history of how that defect was found and fixed, not as
current scores. Everything under defect 4, and every number in the current validation report, is
measured on the leak-free fit-window graph (20 seeds for the scarcity curve, 10 elsewhere).

Full results: [`validation_report.md`](validation_report.md) ·
Build history and where v0 fails: [`v0_process.md`](v0_process.md)
