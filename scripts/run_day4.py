"""Day-4 Task 4 -- the scarcity curve. Does synthetic data substitute for scarce
real data?

Each cell (one k-value x one seed) draws k of the 13 real red-team FIT campaigns,
REFITS the generator distributions on only those k, regenerates a fresh synthetic
corpus from that refit, trains two detectors (real-only and real+synth) and
scores both once on the untouched holdout. The grid is day4.scarcity_k x
day4.n_seeds, with n_seeds overridable on the command line.

THE GRAPH THESE FEATURES COME FROM
----------------------------------
build_context loads graph_fit/aggregates_fit -- bounded to time <= the fit window
-- NOT the full-corpus pair. Unbounded, an edge's weight and therefore its
edge_rarity were computed partly from the evaluation window, i.e. from the future
of the row being scored. That leak was worth real-only AUC-PR 0.9026 -> 0.4122;
every number this file produced before the fix is inflated by it.

WHY THE PER-POINT REFIT IS THE WHOLE TASK
-----------------------------------------
data/derived/synth_auth.csv (10k campaigns) was fit on ALL 13 fit campaigns.
Dropping it into the k=1 arm would smuggle the other 12 campaigns' structure into
the scarce arm and inflate the lift exactly where the headline claim lives. That
corpus is therefore not read anywhere in this file. At k=1 the breadth
distribution collapses to a single value -- that is correct, not a bug: it
faithfully represents having one campaign to learn from.

WHAT IS HELD CONSTANT (so the curve isolates k)
-----------------------------------------------
* synthetic budget: day4.synth_campaigns_per_point campaigns at every k and seed.
  If it tracked k, the experiment would confound "less real data" with "less
  synthetic data".
* evaluation set: all 51 holdout positives against all merged holdout negatives,
  identical for both arms, scored exactly once per fit. Synthetic data appears in
  TRAINING ONLY and is never evaluated on.
* the real-only arm trains on exactly the same k campaigns as the augmented arm,
  not on all 650 events. The only difference between arms is the synth corpus.

NOTHING HERE IS TUNED. No hyperparameter search, no threshold picking, no re-run
with different settings until the curve looks good. GBT params come from
day4.gbt and stay there. A flat curve is a finding, not a number to improve.

Tasks 5, 6 and 9 live at the bottom of this file and reuse load_splits/stack/
build_context so the eval set is bit-identical across every comparison. Each is
its own subcommand -- none of them re-runs the 25-cell sweep, and none of them
rewrites day4_results.json.

Run as a MODULE from the repo root -- `python scripts/run_day4.py` puts scripts/
on sys.path instead of the root and `import src` fails.

     python -m scripts.run_day4 sweep         (Task 4 curve, day4.n_seeds seeds)
     python -m scripts.run_day4 sweep 20      (same, 20 seeds; ~2.8 h)
     python -m scripts.run_day4 --figure-only (rebuild the plot from the JSON)
     python -m scripts.run_day4 baselines     (Task 5: TSTR + 3 baselines)
     python -m scripts.run_day4 ablation      (Task 6: structural-only + importances)
     python -m scripts.run_day4 baserate      (Task 9: test-time base-rate sweep)
     python -m scripts.run_day4 validate      (novelty + SPEC V1 + V2 fidelity)
     python -m scripts.run_day4 all 10        (all four tasks, 10 seeds, one load)

A trailing integer on any subcommand overrides day4.n_seeds. Only the scarcity
curve is worth 20 seeds; the per-seed spread on the leak-free graph is wide
enough that 5 cannot separate the arms.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: no display on the run box
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import average_precision_score

from src.config import load_config
import src.detect as Dt
import src.features as F
import src.parse as P
import src.validate as V
import src.walker as W
import src.writer as Wr

AUTH = list(P.AUTH_COLS)


def load_splits(cfg=None):
    """The ONE definition of the Day-4 train/eval merge. Tasks 5/6/9 reuse it.

    Negatives are benign_* PLUS hard_neg_*: Day-3's benign sampler deleted every
    benign row sitting on a red-team edge, which made positives and negatives
    edge-disjoint by construction and inflated GBT AUC-PR to 0.999. Omitting the
    hard negatives reproduces that artifact. Fit-side files never touch an eval
    set and eval-side files never touch a training set.
    """
    cfg = cfg or load_config()
    p, d3, d4 = cfg["paths"], cfg["day3"], cfg["day4"]

    def negatives(benign_path, hard_path):
        df = pd.concat([pd.read_csv(benign_path), pd.read_csv(hard_path)],
                       ignore_index=True)
        return df[AUTH]

    return {
        "train_pos": pd.read_csv(p["redteam_fit"])[AUTH + ["campaign_id"]],
        "train_neg": negatives(d3["benign_fit"], d4["hard_neg_fit"]),
        "eval_pos": pd.read_csv(p["redteam_holdout"])[AUTH + ["campaign_id"]],
        "eval_neg": negatives(d3["benign_holdout"], d4["hard_neg_holdout"]),
    }


def draw_campaigns(campaign_ids, k, seed):
    """k distinct campaign ids drawn without replacement, seed-reproducible.
    Sorted unique pool first, so the draw cannot depend on row order."""
    ids = np.unique(np.asarray(campaign_ids))
    if k > len(ids):
        raise ValueError(f"k={k} exceeds the {len(ids)} available campaigns")
    picked = np.random.default_rng(seed).choice(ids, size=k, replace=False)
    return sorted(int(i) for i in picked)


def subset_campaigns(pos_df, campaign_ids):
    return pos_df[pos_df["campaign_id"].isin(campaign_ids)]


def stack(pos_frames, neg_frame):
    """One arm's (X, y): positive feature frames stacked over the shared negatives.
    Feature frames, not row frames -- build_features is a pure row-local map, so
    building the 486k negatives once and reusing them is identical to rebuilding
    them per cell, and ~50x cheaper.

    The column check is not decoration: pd.concat takes the UNION of columns and
    fills the gaps with NaN, so stacking a 17-column positive frame over a
    4-column ablated negative frame produces a silently mis-shaped matrix. It
    only surfaced here because sklearn refuses NaN -- a model that tolerated them
    would have returned a plausible, wrong number."""
    frames = list(pos_frames) + [neg_frame]
    cols = list(neg_frame.columns)
    bad = [list(f.columns) for f in frames if list(f.columns) != cols]
    if bad:
        raise ValueError(f"stack() needs identical columns on every frame; "
                         f"negatives have {cols}, got {bad[0]}")
    X = pd.concat(frames, ignore_index=True)
    y = np.concatenate([np.ones(len(f)) for f in pos_frames]
                       + [np.zeros(len(neg_frame))])
    return X, y


def build_context(cfg):
    """Everything constant across all 25 cells, loaded once."""
    t0 = time.time()
    # graph_fit / aggregates_fit, never the full-corpus pair: these feed both
    # build_features (which scores holdout rows) and the walker. See parse._window.
    graph = P.load_graph(cfg["paths"]["graph_fit"])
    with open(cfg["paths"]["aggregates_fit"], "rb") as f:
        agg = pickle.load(f)
    host_users = W.build_host_users(agg["user_host"])
    sp = load_splits(cfg)

    eval_rows = pd.concat([sp["eval_pos"][AUTH], sp["eval_neg"]], ignore_index=True)
    eval_X = F.build_features(eval_rows, graph, host_users)
    eval_y = np.concatenate([np.ones(len(sp["eval_pos"])),
                             np.zeros(len(sp["eval_neg"]))])
    train_neg_X = F.build_features(sp["train_neg"], graph, host_users)

    print(f"context: {len(sp['train_pos'])} train pos / {len(sp['train_neg']):,} "
          f"train neg | {len(sp['eval_pos'])} eval pos / {len(sp['eval_neg']):,} "
          f"eval neg | base rate {eval_y.mean():.6f} | {time.time() - t0:.0f}s")
    return {"cfg": cfg, "graph": graph, "host_users": host_users,
            "hourly": agg["hourly"], "train_pos": sp["train_pos"],
            "train_neg_X": train_neg_X, "eval_X": eval_X, "eval_y": eval_y}


def generate_synth_rows(ctx, dists, seed, naive=False, n=None):
    """Fresh corpus from the k-campaign refit, placed on the timeline and emitted
    as auth rows -- the same generate -> place -> emit sequence as run_day3.

    `naive=True` is Task 5 baseline #2: alpha=beta=0 and credential_bonus=1.0
    flatten the target weights to uniform, and walker's naive flag drops the
    creds-per-campaign restriction and the target-local user preference. Foothold
    selection, the breadth distribution and the V1 real-edge constraint are
    untouched, so campaign size and shape stay controlled and the baseline
    isolates exactly the weighting + credential layers.
    """
    cfg, fo = ctx["cfg"], ctx["cfg"]["fanout"]
    n = n or cfg["day4"]["synth_campaigns_per_point"]
    weighting = (dict(alpha=0.0, beta=0.0, credential_bonus=1.0) if naive else
                 dict(alpha=cfg["alpha"], beta=cfg["beta"],
                      credential_bonus=fo["credential_bonus"]))
    campaigns, stats = W.generate_corpus(
        ctx["graph"], ctx["host_users"], dists, n=n, **weighting,
        min_out_degree=fo["min_foothold_out_degree"],
        max_creds=fo["max_harvest_creds"], seed=seed, naive=naive)
    if not campaigns:
        raise RuntimeError(f"generator produced no campaigns (seed={seed})")

    rng = np.random.default_rng(seed)
    rows, rid = [], 0
    for cid, camp in enumerate(campaigns):
        start = Wr.sample_start_time(
            ctx["hourly"], rng, max_offset=camp[-1][0],
            collection_seconds=cfg["day3"]["collection_seconds"])
        r, lab = Wr.campaign_to_rows(Wr.place_campaign(camp, start), cid, rid)
        for row, l in zip(r, lab):
            row["campaign_id"] = l["campaign_id"]
        rows.extend(r)
        rid += len(r)
    # campaign_id rides along for the validation task (novelty and V1/V2 are all
    # per-campaign). build_features reads columns by name, so the extra column is
    # inert for every other caller.
    return pd.DataFrame(rows)[AUTH + ["campaign_id"]], stats


def scarcity_point(k, seed, ctx):
    """One cell of the curve.
    1) draw k fit campaigns at random (seeded)
    2) refit generator distributions on ONLY those k campaigns
    3) regenerate the synth corpus from that refit
    4) train real-only and real+synth detectors
    5) evaluate both on the untouched holdout
    """
    ids = draw_campaigns(ctx["train_pos"]["campaign_id"], k, seed)
    real = subset_campaigns(ctx["train_pos"], ids)

    dists = W.fit_fanout_distributions(real)          # <- sees only the k campaigns
    synth, gen_stats = generate_synth_rows(ctx, dists, seed=seed * 1000 + k)

    real_X = F.build_features(real[AUTH], ctx["graph"], ctx["host_users"])
    synth_X = F.build_features(synth, ctx["graph"], ctx["host_users"])

    def score(pos_frames):
        X, y = stack(pos_frames, ctx["train_neg_X"])
        return Dt.train_and_eval(X, y, ctx["eval_X"], ctx["eval_y"],
                                 model="gbt", seed=seed)["auc_pr"]

    return {"k": k, "seed": seed, "campaign_ids": ids,
            "n_real_pos": int(len(real)), "n_synth_pos": int(len(synth)),
            "n_synth_campaigns": int(gen_stats["n"]),
            "breadth": [int(b) for b in dists["breadth"]],
            "auc_pr_real": score([real_X]),
            "auc_pr_augmented": score([real_X, synth_X])}


def mean_ci(values):
    """Mean and 95% half-width ACROSS SEEDS. Normal approximation (1.96 * SEM);
    with n=5 that is a rough interval -- reported as such, not as exact."""
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def sweep(ctx, out_path):
    cfg = ctx["cfg"]
    seeds = [cfg["seed"] + i for i in range(cfg["day4"]["n_seeds"])]
    cells, failures = [], []
    total = len(cfg["day4"]["scarcity_k"]) * len(seeds)
    for k in cfg["day4"]["scarcity_k"]:
        for seed in seeds:
            i = len(cells) + len(failures) + 1
            t0 = time.time()
            try:
                cell = scarcity_point(k, seed, ctx)
                cell["seconds"] = round(time.time() - t0, 1)
                cells.append(cell)
                print(f"[{i}/{total}] k={k:2d} seed={seed}  "
                      f"real={cell['auc_pr_real']:.4f}  "
                      f"aug={cell['auc_pr_augmented']:.4f}  "
                      f"lift={cell['auc_pr_augmented'] - cell['auc_pr_real']:+.4f}  "
                      f"({cell['n_real_pos']} real / {cell['n_synth_pos']:,} synth "
                      f"pos, {cell['seconds']}s)", flush=True)
            except Exception as exc:   # decision 7: record, never silently skip
                failures.append({"k": k, "seed": seed, "error": repr(exc)})
                print(f"[{i}/{total}] k={k:2d} seed={seed}  FAILED: {exc!r}",
                      flush=True)
            _dump(out_path, cells, failures, cfg)   # crash-safe: persist each cell
    return cells, failures


def _dump(path, cells, failures, cfg):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"scarcity_k": cfg["day4"]["scarcity_k"],
                   "n_seeds": cfg["day4"]["n_seeds"],
                   "synth_campaigns_per_point": cfg["day4"]["synth_campaigns_per_point"],
                   "gbt": cfg["day4"]["gbt"],
                   "cells": cells, "failures": failures}, f, indent=2)


def summarize(cells, ks):
    """k -> (mean, ci) per arm, aggregated across seeds."""
    out = {}
    for k in ks:
        vals = [c for c in cells if c["k"] == k]
        if not vals:
            continue
        real = [c["auc_pr_real"] for c in vals]
        aug = [c["auc_pr_augmented"] for c in vals]
        out[k] = {"n": len(vals), "real": mean_ci(real), "aug": mean_ci(aug),
                  "paired": paired_test(real, aug)}
    return out


def paired_test(real, aug):
    """Paired t-test on the per-seed lift, plus how many seeds augmentation won.

    Paired because both arms of a cell see the same drawn campaigns, the same
    synthetic seed and the same holdout -- the only difference is whether
    synthetic positives were added. Reported because the per-seed spread is wide
    enough that a mean difference on its own does not establish a direction, and
    a win count says something a mean cannot: whether an effect is consistent or
    a large average over opposite-signed cells."""
    if len(real) < 2:
        return {"n": len(real), "t": None, "p": None, "mean_lift": None,
                "sd_lift": None, "n_seeds_aug_wins": None}
    t, pv = ttest_rel(aug, real)
    d = np.asarray(aug, dtype=float) - np.asarray(real, dtype=float)
    return {"n": len(real), "t": float(t), "p": float(pv),
            "mean_lift": float(d.mean()), "sd_lift": float(d.std(ddof=1)),
            "n_seeds_aug_wins": int((d > 0).sum())}


def plot_curve(summary, path):
    ks = sorted(summary)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, label, color in (("real", "real only", "#1f77b4"),
                              ("aug", "real + synthetic", "#d62728")):
        m = np.array([summary[k][arm][0] for k in ks])
        ci = np.array([summary[k][arm][1] for k in ks])
        ax.plot(ks, m, "o-", color=color, label=label)
        ax.fill_between(ks, m - ci, m + ci, color=color, alpha=0.18, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k = real red-team campaigns available for training")
    ax.set_ylabel("AUC-PR on holdout")
    ax.set_title("Scarcity curve: does synthetic data substitute for scarce real data?\n"
                 "mean over 5 seeds, 95% CI band (normal approx.)", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(summary):
    n = max((s["n"] for s in summary.values()), default=0)
    print(f"\n=== SCARCITY CURVE (mean +/- 95% CI across {n} seeds) ===")
    print(f"{'k':>3}  {'n':>3}  {'real-only':>18}  {'augmented':>18}  {'lift':>9}"
          f"  {'t':>7}  {'p':>6}  {'aug wins':>9}")
    for k in sorted(summary):
        s = summary[k]
        rm, rc = s["real"]
        am, ac = s["aug"]
        pt = s["paired"]
        t = f"{pt['t']:>7.2f}" if pt["t"] is not None else "     --"
        pv = f"{pt['p']:>6.3f}" if pt["p"] is not None else "    --"
        wins = (f"{pt['n_seeds_aug_wins']:>4}/{s['n']:<4}"
                if pt["n_seeds_aug_wins"] is not None else "       --")
        print(f"{k:>3}  {s['n']:>3}  {rm:.4f} +/- {rc:.4f}  {am:.4f} +/- {ac:.4f}  "
              f"{am - rm:>+9.4f}  {t}  {pv}  {wins}")


# =============================================================================
# Shared plumbing for Tasks 5, 6 and 9
# =============================================================================

def seeds_of(cfg):
    return [cfg["seed"] + i for i in range(cfg["day4"]["n_seeds"])]


def full_fit_arm(ctx):
    """The k=13 arm: all 13 real fit campaigns, their feature matrix, and the
    fan-out refit on them. k=13 IS the whole fit split, so there is no draw and
    nothing here depends on the seed -- only generation and the GBT do."""
    real = ctx["train_pos"]
    return (F.build_features(real[AUTH], ctx["graph"], ctx["host_users"]),
            W.fit_fanout_distributions(real))


def _cols(X, cols):
    """Task 6's ablation. build_features is a pure row-local map emitted in a
    fixed order, so slicing the structural columns out of the full matrix is
    identical to rebuilding with groups=("structural",) -- and does not cost a
    second pass over 486k negatives. tests/test_features.py pins that equality."""
    return X if cols is None else X[list(cols)]


def score_arm(ctx, pos_frames, seed, cols=None):
    """Train on the given positive frames over the SHARED negatives, score once
    on the untouched holdout. Returns the full train_and_eval result so callers
    can read feature importances off the fitted model."""
    X, y = stack([_cols(p, cols) for p in pos_frames],
                 _cols(ctx["train_neg_X"], cols))
    return Dt.train_and_eval(X, y, _cols(ctx["eval_X"], cols), ctx["eval_y"],
                             model="gbt", seed=seed)


def sibling_json(results_json, name):
    """Task 5/6/9 results land NEXT TO day4_results.json. The scarcity sweep's
    file is left exactly as Task 4 wrote it -- nothing here reopens it."""
    return str(Path(results_json).with_name(f"day4_{name}.json"))


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def table(rows, arms, title, note=""):
    """mean +/- 95% CI across seeds, one line per arm."""
    print(f"\n=== {title} ===")
    if note:
        print(note)
    width = max(len(label) for _, label in arms)
    for key, label in arms:
        m, ci = mean_ci([r[key] for r in rows])
        print(f"  {label:<{width}}  {m:.4f} +/- {ci:.4f}")


# =============================================================================
# Task 5 -- baselines + TSTR, all scored on the real holdout
# =============================================================================

def rarity_only_auc_pr(ctx):
    """Baseline #3: rank holdout rows by edge_rarity alone. No model, no fit, no
    seed -- one deterministic number, and the control that answers "is a row even
    enough without learning?"."""
    return float(average_precision_score(ctx["eval_y"], ctx["eval_X"]["edge_rarity"]))


BASELINE_ARMS = [
    ("real_only",  "#1 real-fit ceiling (real k=13 only)"),
    ("aug_tuned",  "   real + tuned synthetic"),
    ("aug_naive",  "#2 real + naive synthetic"),
    ("tstr_tuned", "   TSTR, tuned synthetic only"),
    ("tstr_naive", "#2 TSTR, naive synthetic only"),
]


def baselines(ctx, out_path):
    cfg = ctx["cfg"]
    k = max(cfg["day4"]["scarcity_k"])          # 13 = the full fit split
    real_X, dists = full_fit_arm(ctx)
    rarity = rarity_only_auc_pr(ctx)
    print(f"#3 rarity heuristic (edge_rarity, NO training): AUC-PR {rarity:.4f}")

    cells = []
    for seed in seeds_of(cfg):
        t0 = time.time()
        # same refit, same seed, same budget for both generators: the ONLY
        # difference between tuned and naive is the weighting + credential layers
        tuned, t_stats = generate_synth_rows(ctx, dists, seed=seed * 1000 + k)
        naive, n_stats = generate_synth_rows(ctx, dists, seed=seed * 1000 + k,
                                             naive=True)
        tuned_X = F.build_features(tuned, ctx["graph"], ctx["host_users"])
        naive_X = F.build_features(naive, ctx["graph"], ctx["host_users"])
        cell = {"seed": seed,
                "n_synth_pos_tuned": int(len(tuned)),
                "n_synth_pos_naive": int(len(naive)),
                "cap_rate_tuned": t_stats["cap_rate"],
                "cap_rate_naive": n_stats["cap_rate"],
                "real_only":  score_arm(ctx, [real_X], seed)["auc_pr"],
                "tstr_tuned": score_arm(ctx, [tuned_X], seed)["auc_pr"],
                "tstr_naive": score_arm(ctx, [naive_X], seed)["auc_pr"],
                "aug_tuned":  score_arm(ctx, [real_X, tuned_X], seed)["auc_pr"],
                "aug_naive":  score_arm(ctx, [real_X, naive_X], seed)["auc_pr"]}
        cell["seconds"] = round(time.time() - t0, 1)
        cells.append(cell)
        print(f"  seed={seed}  real={cell['real_only']:.4f}  "
              f"aug_tuned={cell['aug_tuned']:.4f}  aug_naive={cell['aug_naive']:.4f}  "
              f"tstr_tuned={cell['tstr_tuned']:.4f}  tstr_naive={cell['tstr_naive']:.4f}"
              f"  ({cell['seconds']}s)", flush=True)
        write_json(out_path, {"k": k, "base_rate": float(ctx["eval_y"].mean()),
                              "rarity_only": rarity, "cells": cells})
    return {"k": k, "rarity_only": rarity, "cells": cells,
            "base_rate": float(ctx["eval_y"].mean())}


def plot_baselines(res, path):
    cells, base = res["cells"], res["base_rate"]
    labels = ["#3 rarity\n(no training)"] + [l.strip() for _, l in BASELINE_ARMS]
    means = [res["rarity_only"]] + [mean_ci([c[k] for c in cells])[0]
                                    for k, _ in BASELINE_ARMS]
    errs = [0.0] + [mean_ci([c[k] for c in cells])[1] for k, _ in BASELINE_ARMS]
    colors = ["#7f7f7f", "#1f77b4", "#d62728", "#ff7f0e", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, means, xerr=errs, color=colors, alpha=0.85, height=0.6)
    ax.axvline(base, color="k", ls=":", lw=1)
    ax.text(base, y[0] + 0.5, f" chance = base rate {base:.5f}", fontsize=8, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC-PR on the real holdout (51 pos / 205,612 neg)")
    ax.set_title("V3 baselines + TSTR, all evaluated on the real holdout\n"
                 "mean over 5 seeds, 95% CI (rarity is training-free: one value)",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Task 6 -- structural-only ablation + feature-importance read-out
# =============================================================================

# None = every feature; STRUCTURAL_COLS = the attribute one-hots dropped
ABLATIONS = (("with attributes", None), ("structural only", F.STRUCTURAL_COLS))


def importances(model, columns):
    return {c: round(float(v), 5)
            for c, v in sorted(zip(columns, model.feature_importances_),
                               key=lambda kv: -kv[1])}


def ablation(ctx, out_path):
    """The primary real-vs-augmented comparison at k=13 ONLY, run twice: once
    with the full feature set and once with the attribute one-hots dropped.
    k=13 alone, not the whole 25-cell sweep -- the diagnosis reads the same at
    every k and a second full sweep is ~2h of fits for no extra information.
    """
    cfg = ctx["cfg"]
    k = max(cfg["day4"]["scarcity_k"])
    real_X, dists = full_fit_arm(ctx)
    cells = []
    for seed in seeds_of(cfg):
        t0 = time.time()
        synth, _ = generate_synth_rows(ctx, dists, seed=seed * 1000 + k)
        synth_X = F.build_features(synth, ctx["graph"], ctx["host_users"])
        cell = {"seed": seed, "n_synth_pos": int(len(synth))}
        for name, cols in ABLATIONS:
            r = score_arm(ctx, [real_X], seed, cols)
            a = score_arm(ctx, [real_X, synth_X], seed, cols)
            cell[f"real::{name}"] = r["auc_pr"]
            cell[f"aug::{name}"] = a["auc_pr"]
            if cols is None:   # importance read-out from the full-feature models
                names = list(ctx["eval_X"].columns)
                cell["importance_real"] = importances(r["model"], names)
                cell["importance_aug"] = importances(a["model"], names)
        cell["seconds"] = round(time.time() - t0, 1)
        cells.append(cell)
        print("  seed={}  ".format(seed) + "  ".join(
            f"{arm}/{name}={cell[f'{arm}::{name}']:.4f}"
            for name, _ in ABLATIONS for arm in ("real", "aug"))
            + f"  ({cell['seconds']}s)", flush=True)
        write_json(out_path, {"k": k, "cells": cells})
    return {"k": k, "cells": cells}


def mean_importance(cells, key):
    """Importances averaged across seeds; the models differ only by random_state,
    so the mean is the stable read-out."""
    keys = cells[0][key].keys()
    return dict(sorted(((c_, float(np.mean([cell[key][c_] for cell in cells])))
                        for c_ in keys), key=lambda kv: -kv[1]))


def plot_ablation(res, path):
    cells = res["cells"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    names = [n for n, _ in ABLATIONS]
    x = np.arange(len(names))
    for i, (arm, label, color) in enumerate((("real", "real only", "#1f77b4"),
                                             ("aug", "real + synthetic", "#d62728"))):
        m = [mean_ci([c[f"{arm}::{n}"] for c in cells])[0] for n in names]
        e = [mean_ci([c[f"{arm}::{n}"] for c in cells])[1] for n in names]
        ax1.bar(x + (i - 0.5) * 0.35, m, 0.35, yerr=e, label=label, color=color,
                alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("AUC-PR on holdout")
    ax1.set_title(f"Task 6 ablation at k={res['k']}\n(mean over 5 seeds, 95% CI)",
                  fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    imp_r = mean_importance(cells, "importance_real")
    imp_a = mean_importance(cells, "importance_aug")
    top = list(imp_r)[:8]
    y = np.arange(len(top))[::-1]
    ax2.barh(y + 0.18, [imp_r[c] for c in top], 0.36, color="#1f77b4",
             label="real only", alpha=0.85)
    ax2.barh(y - 0.18, [imp_a[c] for c in top], 0.36, color="#d62728",
             label="real + synthetic", alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels(top, fontsize=8)
    ax2.set_xlabel("GBT feature importance (mean over seeds)")
    ax2.set_title("Where the fitted models put their mass\n"
                  "(top 8 by the real-only model)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Task 9 -- test-time base-rate sweep
# =============================================================================

def downsample_eval(eval_y, rate, seed):
    """Row indices for a holdout re-balanced so positives are `rate` of the rows.

    ONLY negatives are dropped: all 51 real holdout positives survive every point,
    or the sweep would stop measuring "how does precision behave as the haystack
    shrinks" and start measuring a different, smaller set of attacks.
    rate=None returns every row untouched -- the natural ~1:4000 floor.
    """
    idx = np.arange(len(eval_y))
    if rate is None:
        return idx
    pos, neg = idx[eval_y == 1], idx[eval_y == 0]
    n_neg = int(round(len(pos) * (1.0 - rate) / rate))
    if n_neg > len(neg):
        raise ValueError(
            f"base rate {rate} needs {n_neg:,} negatives but the holdout has "
            f"only {len(neg):,}; {len(pos) / (len(pos) + len(neg)):.6g} is the floor")
    keep = np.random.default_rng(seed).choice(neg, size=n_neg, replace=False)
    return np.sort(np.concatenate([pos, keep]))


def baserate_sweep(ctx, out_path):
    """One real-only k=13 fit per seed -- the best honest model -- scored ONCE on
    the full holdout. The sweep then subsets that score vector, so every base-rate
    point sees the identical model and the identical 51 positives; only the benign
    haystack shrinks. Ranking is unchanged by subsetting, so this is exactly
    equivalent to re-scoring each down-sampled set, minus four redundant fits.
    """
    cfg = ctx["cfg"]
    rates = cfg["day4"]["holdout_base_rates"]
    real_X, _ = full_fit_arm(ctx)
    cells = []
    for seed in seeds_of(cfg):
        t0 = time.time()
        model = score_arm(ctx, [real_X], seed)["model"]
        scores = model.predict_proba(ctx["eval_X"])[:, 1]
        cell = {"seed": seed}
        for rate in rates:
            idx = downsample_eval(ctx["eval_y"], rate, seed)
            y = ctx["eval_y"][idx]
            cell[str(rate)] = {
                "auc_pr": float(average_precision_score(y, scores[idx])),
                "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
                "achieved_base_rate": float(y.mean())}
        cell["seconds"] = round(time.time() - t0, 1)
        cells.append(cell)
        print(f"  seed={seed}  " + "  ".join(
            f"{r}:{cell[str(r)]['auc_pr']:.4f}" for r in rates)
            + f"  ({cell['seconds']}s)", flush=True)
        write_json(out_path, {"holdout_base_rates": rates, "cells": cells})
    return {"holdout_base_rates": rates, "cells": cells}


def report_baserate(res):
    """Rarest first: the natural floor is the honest operating point."""
    rates = sorted(res["holdout_base_rates"],
                   key=lambda r: (r is not None, r))   # None (floor) leads
    cells = res["cells"]
    print("\n=== TASK 9: HOLDOUT BASE-RATE SWEEP (real-only k=13, rarest first) ===")
    print(f"{'target':>10}  {'ratio':>10}  {'pos':>4}  {'negatives':>10}  "
          f"{'achieved':>10}  {'AUC-PR (mean +/- CI)':>22}")
    for r in rates:
        c0 = cells[0][str(r)]
        m, ci = mean_ci([c[str(r)]["auc_pr"] for c in cells])
        achieved = c0["achieved_base_rate"]
        print(f"{('natural' if r is None else r):>10}  "
              f"{'1:%d' % round(1 / achieved - 1):>10}  {c0['n_pos']:>4}  "
              f"{c0['n_neg']:>10,}  {achieved:>10.6f}  {m:>13.4f} +/- {ci:.4f}")


def plot_baserate(res, path):
    rates = res["holdout_base_rates"]
    cells = res["cells"]
    x = [cells[0][str(r)]["achieved_base_rate"] for r in rates]
    m = np.array([mean_ci([c[str(r)]["auc_pr"] for c in cells])[0] for r in rates])
    ci = np.array([mean_ci([c[str(r)]["auc_pr"] for c in cells])[1] for r in rates])
    order = np.argsort(x)
    x = np.array(x)[order]; m, ci = m[order], ci[order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, m, "o-", color="#1f77b4", label="real-only k=13 detector")
    ax.fill_between(x, m - ci, m + ci, color="#1f77b4", alpha=0.18, linewidth=0)
    ax.plot(x, x, "k:", lw=1, label="chance (AUC-PR = base rate)")
    ax.set_xscale("log")
    ax.set_xlabel("holdout base rate (positives / all rows)")
    ax.set_ylabel("AUC-PR")
    ax.set_title("Task 9: test-time base-rate sweep\n"
                 "all 51 real positives kept at every point; only benign is "
                 "down-sampled", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# entry points
# =============================================================================

def run_sweep(cfg):
    d4 = cfg["day4"]
    out_json = d4["results_json"]
    fig_path = d4["figures_dir"].rstrip("/") + "/scarcity_curve.png"

    if "--figure-only" in sys.argv:
        with open(out_json, encoding="utf-8") as f:
            saved = json.load(f)
        cells, failures = saved["cells"], saved["failures"]
    else:
        ctx = build_context(cfg)
        cells, failures = sweep(ctx, out_json)

    summary = summarize(cells, d4["scarcity_k"])
    report(summary)
    if failures:
        print(f"\n[!] {len(failures)} cell(s) FAILED:")
        for f_ in failures:
            print(f"    k={f_['k']} seed={f_['seed']}: {f_['error']}")
    plot_curve(summary, fig_path)
    print(f"\nfigure -> {fig_path}\nresults -> {out_json}")

    missing = [k for k in d4["scarcity_k"] if k not in summary]
    ok = not missing and not failures
    print(f"[{'x' if ok else ' '}] both arms present at all {len(d4['scarcity_k'])} "
          f"k-values (missing={missing}, failures={len(failures)})")


def run_baselines(ctx, figures, out_json):
    res = baselines(ctx, out_json)
    table(res["cells"], BASELINE_ARMS,
          f"TASK 5: V3 BASELINES + TSTR at k={res['k']} (real holdout, "
          f"mean +/- 95% CI over 5 seeds)",
          f"  #3 rarity heuristic, no training, no seed:  {res['rarity_only']:.4f}\n"
          f"  chance (= holdout base rate):               {res['base_rate']:.6f}")
    fig = f"{figures}/baselines_tstr.png"
    plot_baselines(res, fig)
    print(f"\nfigure -> {fig}\nresults -> {out_json}")


def run_ablation(ctx, figures, out_json):
    res = ablation(ctx, out_json)
    arms = [(f"{arm}::{name}", f"{label} / {name}")
            for name, _ in ABLATIONS
            for arm, label in (("real", "real only"), ("aug", "real + synthetic"))]
    table(res["cells"], arms,
          f"TASK 6: STRUCTURAL-ONLY ABLATION at k={res['k']} "
          f"(mean +/- 95% CI over 5 seeds)")
    for key, label in (("importance_real", "real-only"), ("importance_aug", "augmented")):
        imp = mean_importance(res["cells"], key)
        print(f"\n  GBT feature importance, {label} model (top 6, mean over seeds):")
        for c, v in list(imp.items())[:6]:
            print(f"    {c:<22} {v:.4f}")
        struct = sum(v for c, v in imp.items() if c in F.STRUCTURAL_COLS)
        print(f"    -> structural mass {struct:.4f} / attribute mass {1 - struct:.4f}")
    fig = f"{figures}/ablation_structural.png"
    plot_ablation(res, fig)
    print(f"\nfigure -> {fig}\nresults -> {out_json}")


def run_baserate(ctx, figures, out_json):
    res = baserate_sweep(ctx, out_json)
    report_baserate(res)
    fig = f"{figures}/baserate_sweep.png"
    plot_baserate(res, fig)
    print(f"\nfigure -> {fig}\nresults -> {out_json}")


def dst_indeg_quantiles(synth, fit, graph):
    """p25/50/75 of per-EVENT target in-degree, both sides.

    The single most legible statement of the generator's failure -- the JS
    divergence says the distributions differ, these three numbers say how: the
    generator aims at hubs the real red team avoided. Reported alongside the JS
    so the direction of the gap is in the artifact, not only in prose.
    """
    indeg = dict(graph.in_degree())

    def q(df):
        v = df["dst_computer"].map(indeg).fillna(0).to_numpy(dtype=float)
        return {str(p): float(np.percentile(v, p)) for p in (25, 50, 75)}

    return {"synth": q(synth), "real": q(fit)}


def run_validate(ctx, figures, out_json):
    """Novelty + SPEC V1 + V2 fidelity on a fresh full-size corpus.

    src/validate.py had no caller: the novelty and V1/V2 numbers in the report
    were produced by an ad-hoc script that was never committed, so nothing in the
    repo reproduced them. This task is that missing caller. It regenerates the
    corpus in-process from the frozen fit split rather than reading Day-3's CSVs,
    so the statistics always describe the generator as currently configured --
    the stale CSVs were built against the pre-t_hi full-corpus graph.
    """
    cfg = ctx["cfg"]
    n = cfg["fanout"]["n_campaigns"]
    fit = ctx["train_pos"]
    dists = W.fit_fanout_distributions(fit)
    synth, _ = generate_synth_rows(ctx, dists, seed=cfg["seed"], n=n)
    print(f"corpus: {len(synth):,} rows / {synth['campaign_id'].nunique():,} campaigns")

    nov = V.novelty_metrics(V.campaign_edge_sets(synth), V.campaign_edge_sets(fit),
                            near_dup_threshold=cfg["day4"]["near_dup_threshold"])
    v1 = V.v1_assertions(synth, ctx["graph"],
                         collection_seconds=cfg["day3"]["collection_seconds"])
    v2 = V.v2_metrics(synth, fit, ctx["graph"])

    mx = nov["max_jaccard"]
    overlap = nov["n_synth_edges"] - round(nov["edge_novelty_rate"] * nov["n_synth_edges"])
    print()
    print("NOVELTY")
    print(f"  max-Jaccard median={np.median(mx):.4f} "
          f"p95={np.percentile(mx, 95):.4f} p99={np.percentile(mx, 99):.4f}")
    print(f"  exactly 0: {int((mx == 0).sum()):,} / {len(mx):,}")
    print(f"  pct_near_duplicate={nov['pct_near_duplicate']:.4f}  "
          f"edge_novelty_rate={nov['edge_novelty_rate']:.4f}")
    print(f"  distinct edges synth={nov['n_synth_edges']:,} "
          f"fit={nov['n_fit_edges']:,} overlap={overlap:,}")

    print()
    print("V1 (SPEC hard constraints)")
    for name, res in v1.items():
        if isinstance(res, dict) and "status" in res:
            # the credential check is status "n/a" and carries no count
            print(f"  [{res['status']:>3}] {name:<34} "
                  f"violations={res.get('n_violations', '-')}")

    print()
    print("V2 (fidelity; only 13 real campaigns -- read statistic, not p)")
    for name, d in v2["ks"].items():
        print(f"  {name:<20} KS={d['statistic']:.4f}  "
              f"median synth={d['median_synth']:.2f} real={d['median_real']:.2f}")
    print(f"  dst_in_degree JS   ={v2['dst_in_degree_js']:.4f}")

    write_json(out_json, {
        "n_campaigns": n,
        "novelty": {k: v for k, v in nov.items() if k != "max_jaccard"},
        "max_jaccard_quantiles": {q: float(np.percentile(mx, q)) for q in (50, 95, 99)},
        "n_exact_zero_jaccard": int((mx == 0).sum()),
        "v1": v1,
        "v2": {"ks": v2["ks"], "dst_in_degree_js": v2["dst_in_degree_js"],
               "dst_in_degree_quantiles": dst_indeg_quantiles(synth, fit, ctx["graph"])}})
    V.plot_v2_figures(v2, figures, graph=ctx["graph"], synth_df=synth, fit_df=fit)
    V.plot_hourly(synth, ctx["hourly"], figures)
    print()
    print(f"figures -> {figures}/")
    print(f"results -> {out_json}")


TASKS = {"baselines": run_baselines, "ablation": run_ablation,
         "baserate": run_baserate, "validate": run_validate}


def main():
    cfg = load_config()
    figures = cfg["day4"]["figures_dir"].rstrip("/")
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    cmd = args[0] if args else "sweep"
    # Optional seed-count override: `run_day4.py sweep 20`. The scarcity curve is
    # the headline and needs the power; the other three tasks do not earn 4x the
    # compute, so the count belongs on the command line, not pinned in config.
    if len(args) > 1:
        cfg["day4"]["n_seeds"] = int(args[1])
    print(f"n_seeds={cfg['day4']['n_seeds']}")

    if cmd == "sweep":
        return run_sweep(cfg)
    if cmd == "all":
        chosen = list(TASKS)
    elif cmd in TASKS:
        chosen = [cmd]
    else:
        raise SystemExit(f"unknown command {cmd!r}; "
                         f"expected sweep, all, or one of {sorted(TASKS)}")

    ctx = build_context(cfg)     # loaded once, reused by every chosen task
    for name in chosen:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}", flush=True)
        TASKS[name](ctx, figures, sibling_json(cfg["day4"]["results_json"], name))


if __name__ == "__main__":
    main()
