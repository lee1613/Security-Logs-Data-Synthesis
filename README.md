# Synthetic Lateral-Movement Log Synthesis

Generating synthetic red-team authentication events from the LANL auth corpus, and measuring
whether that synthetic data is actually **useful** — not whether it looks plausible.

Spec: [`SPEC_synthetic_lateral_movement.md`](SPEC_synthetic_lateral_movement.md) ·
Results: [`docs/validation_report.md`](docs/validation_report.md)

## Headline finding

**The synthetic corpus is worse than nothing as training augmentation.** Across a 25-cell scarcity
sweep it costs **0.39–0.60 AUC-PR at every level of real-data scarcity**, and a detector trained
only on synthetic data is near-useless on real attacks (TSTR 0.06–0.12).

The failure is **generative**, not structural or attributional. The corpus is genuinely novel — 98%
of its edges were never traversed by the real red team, with no memorization — but its per-event
feature distribution does not match the real attack, and at a ~20:1 synthetic-to-real ratio it
dominates the positive class and pulls the detector onto a coarse tooling attribute (`NTLM`).

The single most useful number required no machine learning at all: **ranking rows by edge rarity
alone scores 0.530 — 59% of the fully trained detector.** That is the bar any synthesis work has to
clear.

## The insight worth carrying

> The synthetic data faithfully reproduces the actor's business-hours tradecraft, but timing is
> **deliberately excluded** from the detector — **a naive "flag out-of-hours activity" rule would
> miss this red team entirely.**

**Behavioral fidelity and detection utility are separate axes.** A feature can be behaviorally real
and detection-useless. Hour-of-day is exactly that here: benign traffic peaks in the same hours, and
there are zero off-hours attack examples to learn from. Reproducing an attacker's rhythm is not the
same as reproducing what makes them findable.

Three more framing corrections this project had to make about itself:

- **Fan-out ≠ multi-hop.** The real data shows star-shaped fan-out from a foothold, not the linear
  multi-hop chains the spec originally assumed. The generator was rebuilt around what the data
  actually showed.
- **Attribute constancy is tool-specific.** 100% `NTLM`/`Network` is this red team's toolkit, not a
  property of lateral movement. A detector riding it has learned one operator's tooling.
- **Within-operator transfer only.** Same team, same toolkit, one engagement. A different attacker
  is not demonstrated.

## Pipeline

| Day | What it does | Entry point |
|---|---|---|
| 1 | Aggregate the ~1B-row auth corpus; build the host graph; recover red-team events | `scripts/run_day1.py` |
| 2 | Fit fan-out distributions; generate campaigns on real graph edges | `scripts/run_day2.py` |
| 3 | Place campaigns on the timeline; emit labeled corpora at three base rates | `scripts/run_day3.py` |
| 4 | Features, detector, scarcity curve, baselines, ablation, validation | `scripts/run_day4.py` |

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 98 tests
python scripts/run_day4.py          # scarcity sweep (~90 min)
python scripts/run_day4.py all      # baselines, ablation, base-rate sweep
```

## Modules

| File | Responsibility |
|---|---|
| `src/parse.py` | DuckDB aggregation over the raw corpus; graph construction; red-team recovery |
| `src/walker.py` | Fan-out campaign generator (fits distributions, generates on real edges) |
| `src/writer.py` | Timeline placement and row/label emission |
| `src/benign.py` | Windowed benign sampling for negatives |
| `src/features.py` | Row → feature vector; leakage-safe, pure row-local map |
| `src/detect.py` | Class-weighted train/eval harness returning AUC-PR |
| `src/validate.py` | V1 assertions, V2 distributional tests, novelty metrics |

## Two evaluation defects found and fixed

Both are documented in full in the validation report; both are the kind that produce
*better-looking* numbers, which is why they were worth chasing.

1. **Edge-disjoint classes.** The benign sampler excluded every row on a red-team edge, so positives
   and negatives shared zero edges and the detector could learn a hole in feature space rather than
   the attack. Worth a spurious **0.999 → 0.894** once repaired. Fixed by
   `scripts/recover_hard_negatives.py`.
2. **Non-reproducible generation.** Credential harvesting indexed a `set` of strings, whose order
   Python randomizes per process, so a fixed seed produced a different corpus every run. Fixed with
   `sorted()`, guarded by a cross-process regression test.

## Honest-outcome policy

A negative result with clean methodology is a legitimate outcome, and this project produced one.
Nothing in the report was tuned: no hyperparameter search, no threshold picking, no re-running until
the curve looked better. Where a check failed, it is reported as failed.
