# Day 4 — Augmentation & Evaluation — Implementation Plan (THIN)

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
>
> **This plan is deliberately THIN.** It specifies *interfaces, decisions, and test strategy* — not full code bodies. This is a correction of the Day-3 plan, which pre-wrote every line and reduced execution to transcription. Here the implementer writes the code; the plan constrains *what it must do and must not do*.

**Goal:** Answer three genuinely open questions about the Day-3 synthetic data product:
1. **Augmentation lift** — does synthetic data substitute for scarce real data? (headline)
2. **Shortcut ablation** — is the detectable signal real *structure*, or the NTLM attribute artifact?
3. **Novelty / memorization** — does synth add coverage beyond the 13 fit campaigns, or replay them?

**Framing (do not drift from this).** The detector is a **measuring instrument for synthetic-data quality**, NOT a detection-science contribution. We are *not* testing whether lateral movement is graph/sequential — the field knows it is. Absolute detection AUC is semi-predictable (the rarity baseline will likely be strong; trained lift modest). Do **not** headline detection performance. SPEC §5 line 332 stands: a negative result with clean methodology is a legitimate outcome.

**Scope:** Day 4 only. All decisions below were settled at the Day-3→Day-4 grilling checkpoint (2026-08-11). **Do not relitigate them.**

---

## Grounded facts (verified — trust, don't re-derive)

| Fact | Value |
|---|---|
| Real holdout positives | **51 events / 6 campaigns** — the binding statistical constraint, cannot grow |
| Real fit positives | 650 events / **13 campaigns** |
| Benign available | `benign_fit.csv` 483,764 · `benign_holdout.csv` 204,228 |
| Synth corpus (all-13 fit) | `synth_auth.csv` 144,228 events / 10k campaigns, 0 edge violations |
| Holdout base-rate floor | **~1:4000** (51 pos / 204k benign) |
| Categorical fields | constant `(NTLM, Network, LogOn, Success)`, `dst_user == src_user` |
| Day-2 generation cost | ~34s per 10k campaigns (so per-point regeneration is cheap) |
| Graph / creds source | `graph.pkl` + `aggregates["user_host"]` — built from **benign** traffic |

**Leakage rule (absolute):** every feature is computed from the **benign graph + fit split only**. The holdout is touched exactly once, at evaluation. Any feature derived from holdout data is a bug.

---

## Settled decisions (from the grilling checkpoint — encoded, not open)

- **Unit = per-row.** No edge- or campaign-level modelling. Per-event has a known ceiling → limitations, not hidden.
- **Two headlines:** augmentation scarcity curve leads; TSTR + 3 baselines underneath (satisfies SPEC V3).
- **Detector:** shallow regularized GBT (depth ~3) primary; logistic regression once, for legibility only.
- **Hour-of-day is NOT a detection feature.** Near-zero discriminator (benign peaks in the same hours; zero off-hours attack examples). Retained only as a V2 distributional figure.
- **Benign as-is.** No reservoir fix. Holdout floor ~1:4000 accepted.
- **Base rate is a TEST-TIME axis** — swept by down-sampling holdout benign, always keeping all 51 positives. Train with **class weighting**. Day-3 `mixed_*` corpora are a training-balance robustness check only.
- **Never evaluate on synthetic data.** Every reported number is measured on the real holdout.
- **Per-point synth regeneration** on the scarcity curve (see Task 4) — non-negotiable.

---

## File structure

| File | Responsibility |
|---|---|
| `config.yaml` | Add a `day4:` block (scarcity k-values, seed count, base-rate sweep, GBT params, near-dup threshold). |
| `src/features.py` | Row → feature-vector builder. Leakage-safe (benign graph + fit only). Structural / attribute feature groups separable for the ablation. |
| `src/detect.py` | Train/eval harness: fit a detector on a given training set, score a given eval set, return AUC-PR + PR curve. Class-weighted. Model-agnostic (GBT or logistic). |
| `src/validate.py` | V1 assertions, V2 distributional tests, novelty/memorization metrics. |
| `src/walker.py` | **Modify:** add a naive-mode flag or thin variant for baseline #2 (see Task 5). No change to existing behaviour. |
| `scripts/run_day4.py` | Orchestrate: V1/V2 → scarcity curve → ablation → TSTR table + baselines → base-rate sweep → emit report data + figures. |
| `tests/test_features.py` | Feature correctness + **leakage guards**. |
| `tests/test_detect.py` | Harness correctness on synthetic fixtures. |
| `tests/test_validate.py` | V1/V2/novelty metric correctness. |
| `docs/validation_report.md` | The deliverable. |

**Mechanics:** branch `day4-augmentation-eval`; merge at the gate. Report ships as repo markdown **and** a published artifact.

---

## Interfaces (contracts the implementer must honour)

Signatures are **binding**; bodies are the implementer's.

```
build_features(rows_df, graph, host_users, *, groups=("structural","attribute")) -> DataFrame
    # rows_df: any auth rows in AUTH_COLS order (synth, real red-team, or benign)
    # returns one feature row per input row, index-aligned, deterministic
    # groups selects which feature families to emit -> this is the ablation switch

train_and_eval(train_X, train_y, eval_X, eval_y, *, model="gbt", seed) -> dict
    # class-weighted fit; returns {"auc_pr": float, "pr_curve": (precision, recall),
    #                              "model": fitted_model}

scarcity_point(k, seed, ...) -> dict
    # 1) draw k fit campaigns at random (seeded)
    # 2) refit generator distributions on ONLY those k campaigns
    # 3) regenerate synth corpus from that refit
    # 4) train real-only and real+synth detectors
    # 5) evaluate both on the untouched holdout
    # returns {"k","seed","auc_pr_real","auc_pr_augmented"}

novelty_metrics(synth_campaigns, fit_campaigns, *, near_dup_threshold=0.5) -> dict
    # {"max_jaccard": array, "pct_near_duplicate": float, "edge_novelty_rate": float}
```

**Feature families (exact):**
- `structural` — `edge_rarity` (−log benign edge frequency; unseen edge = max), `dst_in_degree`, `src_out_degree`, `credential_novelty` (binary: is this `src_user` seen on this `dst_computer` in benign history), `n_hosts_for_cred`
- `attribute` — `auth_type`, `logon_type` one-hot **(the shortcut — dropped in the ablation)**
- **Excluded entirely:** `hour_of_day` (decision above), `success`/`auth_orientation` (constant on both sides → no information)

---

## Tasks

### Task 1: `day4` config block
Add scarcity k-values `[1,2,4,8,13]`, `n_seeds: 5`, holdout base-rate sweep `[0.01, 0.001, null]` (`null` = natural floor, all 204k benign), GBT params (shallow: depth ~3, modest n_estimators, regularized), `near_dup_threshold: 0.5`, output paths.
**Exit:** parses; values readable via `load_config()`.

### Task 2: `src/features.py` + leakage guards (TDD)
Build the feature matrix per the families above.
**Required tests — the leakage guards are the point:**
- feature values for a given row are **identical** whether or not holdout rows are present in the input (proves no holdout-derived statistics)
- an edge absent from the benign graph gets maximum `edge_rarity` (no crash, no silent zero)
- `groups=("structural",)` returns *only* structural columns — the ablation switch actually works
- deterministic: same input → same output
- known-value tests on a tiny hand-built graph
**Exit:** all pass; no holdout path reachable from the feature builder.

### Task 3: `src/detect.py` harness (TDD)
Class-weighted fit + score; return AUC-PR and PR curve; support `model="gbt"|"logistic"`; seeded.
**Tests:** separable fixture → AUC-PR ≈ 1.0; pure-noise fixture → AUC-PR ≈ base rate; same seed → identical result; class weighting actually applied (imbalanced fixture doesn't collapse to all-negative).
**Exit:** all pass.

### Task 4: Scarcity curve — **the headline** (TDD on mechanics, then run)
`scarcity_point(k, seed)` per the contract above. 5 k-values × 5 seeds = 25 cells; each cell regenerates synth.
**The leakage trap this task exists to avoid:** the existing 10k synth corpus was fit on **all 13** fit campaigns. Adding it at k=1 would smuggle the other 12 campaigns' structure into the scarce arm and **inflate the lift exactly where the headline claim lives**. Therefore: refit generator distributions on **only the k drawn campaigns**, regenerate, then augment. At k=1 the breadth distribution degenerates to a single value — that is **correct**, it faithfully represents having one campaign to learn from.
**Tests:** the refit at k uses only the k campaigns' events (assert on the distributions, not the output); random subset is seed-reproducible; holdout is never in any training set.
**Output:** mean ± 95% CI per k, both arms; figure with CI bands.
**Exit:** curve produced; real-only and augmented arms both present at all 5 k.

### Task 5: Baselines + TSTR table
- **#1 real-fit ceiling** — train on the 13 fit campaigns' real events. No build.
- **#3 rarity heuristic** — rank rows by `edge_rarity` alone, no training, compute AUC-PR. No build. *This is the "is a row even enough without learning?" control — report it prominently.*
- **#2 naive uniform fan-out** — thin `walker.py` variant: `alpha=0, beta=0, credential_bonus=1.0`, **no** creds-per-campaign restriction, users assigned at random from the compromised pool. Keeps real footholds, real breadth distribution, and the V1 edge constraint (so campaign size/shape is controlled) — isolates exactly the weighting + credential layers.
**Exit:** V3 TSTR table complete with all three baselines, all evaluated on the real holdout.

### Task 6: Structural-only ablation
Re-run the primary comparison with `groups=("structural",)` — attribute features dropped.
**Interpretation:** if detection survives, the signal is real structure, not the NTLM shortcut. If it collapses, the detector was riding a synthetic artifact — **report that plainly**; it is a finding, not a failure.
**Exit:** with/without-attribute numbers side by side.

### Task 7: Novelty / memorization
Per the `novelty_metrics` contract, on the **full 10k corpus (fit on all 13)** vs the 13 fit campaigns.
Note the nuance in the writeup: every synth edge is already a real graph edge (V1 constraint), so "novel" means **a real edge the red team did not traverse in fit** — precisely the extra coverage synth contributes.
**Read jointly with the scarcity lift:** lift + high novelty ⇒ genuine generalization · lift + low novelty ⇒ discount as memorization · no lift + high novelty ⇒ honest negative (synth explores real edges that aren't discriminative).
**Exit:** max-Jaccard distribution, % near-duplicate, edge novelty rate.

### Task 8: V1 / V2 (SPEC exit criteria)
**V1 assertions:** timestamps strictly increasing within campaign & all Δt>0 · no campaign beyond the 58-day window · every `(src,dst)` real (already verified Day-3, 0 violations — re-assert) · *"no credential used before acquisition" is **N/A by architecture*** (fan-out has no hops; all credentials are harvested at the foothold at t=0) — **state this plainly, do not fake an assertion**.
**V2:** KS test on breadth / inter-event Δt / creds-per-campaign (the last one verifies the Day-3 gap-#2 fix landed) · JS divergence on dst in-degree, target <0.1 · overlaid histograms · hourly synth-vs-real figure as the **Caveat B footnote** · note that constant-categorical JS≈0 proves nothing (SPEC line 278).
**Exit:** V1 assertions pass; V2 figures produced for every listed field.

### Task 9: Base-rate sweep (test-time)
Down-sample **holdout benign** to hit 1:100, 1:1000, and the natural ~1:4000 floor — **always keeping all 51 positives**. Report AUC-PR at each; **lead with the rarest**.
**Exit:** three-point sweep table on real holdout data.

### Task 10: `validation_report.md` + README insights
**Report order:** headline scarcity curve → novelty/memorization → structural-only ablation → V3 TSTR table with 3 baselines → base-rate sweep → V1/V2 → **limitations**.
**README insight (from the gate, must appear):** the synthetic data faithfully reproduces the actor's business-hours tradecraft, but timing is deliberately **excluded** from the detector — **a naive "flag out-of-hours activity" rule would miss this red team entirely.** Behavioral fidelity and detection utility are separate axes; a feature can be behaviorally real and detection-useless.
Carry forward the framing corrections already banked in the phase-gate doc: fan-out ≠ multi-hop · don't dress inference as fact · within-operator transfer ceiling · attribute constancy is tool-specific.

**Limitations section — first-class, not an afterthought:**
per-event altitude has a known ceiling · within-operator transfer only (same red team, same toolkit — a different attacker is NOT demonstrated) · cred→target assignments are a benign-co-occurrence proxy (gap #1) · single-foothold vs multi-origin reality (gap #5) · benign label noise (`redteam.txt` is detection-incomplete; "benign" may hide undetected attacks) · **51 holdout positives → wide CIs** · holdout base-rate floor ~1:4000 · business-hour drift 0.55 (Caveat B) · broad foothold pool is deliberate, not an oversight (gap #3).

**Exit:** report complete with figures; published as an artifact; no claim stronger than *"reproduces THIS operator's detectable structure."*

---

## Day-4 exit criteria

- [ ] V1 assertions pass (3 real + 1 documented N/A)
- [ ] V2 figures produced for every listed field
- [ ] V3 TSTR table complete with all three baselines
- [ ] Scarcity curve with 5 k-values × 5 seeds, mean ± 95% CI, **synth regenerated per point**
- [ ] Structural-only ablation reported side by side
- [ ] Novelty metrics reported and read jointly with lift
- [ ] Base-rate sweep on holdout, leading with the rarest
- [ ] Limitations section states plainly where the data falls short

**Honest-outcome clause:** if the augmentation lift is small or the rarity baseline matches the trained detector, **report it**. Diagnose whether the limit is structural (per-event altitude), attributional (shortcut), or generative (synth doesn't carry the signal) — and say which. Do not tune until pretty.
