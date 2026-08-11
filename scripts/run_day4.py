"""Day-4 Task 4 -- the scarcity curve. Does synthetic data substitute for scarce
real data?

Each of the 25 cells (5 k-values x 5 seeds) draws k of the 13 real red-team FIT
campaigns, REFITS the generator distributions on only those k, regenerates a
fresh synthetic corpus from that refit, trains two detectors (real-only and
real+synth) and scores both once on the untouched holdout.

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

Run: python scripts/run_day4.py            (~60-90 min; per-cell progress printed)
     python scripts/run_day4.py --figure-only   (rebuild the plot from the JSON)
"""
import json
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")           # headless: no display on the run box
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
import src.detect as Dt
import src.features as F
import src.parse as P
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
    them per cell, and ~50x cheaper."""
    X = pd.concat(list(pos_frames) + [neg_frame], ignore_index=True)
    y = np.concatenate([np.ones(len(f)) for f in pos_frames]
                       + [np.zeros(len(neg_frame))])
    return X, y


def build_context(cfg):
    """Everything constant across all 25 cells, loaded once."""
    t0 = time.time()
    graph = P.load_graph(cfg["paths"]["graph"])
    with open(cfg["paths"]["aggregates"], "rb") as f:
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


def generate_synth_rows(ctx, dists, seed):
    """Fresh corpus from the k-campaign refit, placed on the timeline and emitted
    as auth rows -- the same generate -> place -> emit sequence as run_day3."""
    cfg, fo = ctx["cfg"], ctx["cfg"]["fanout"]
    n = cfg["day4"]["synth_campaigns_per_point"]
    campaigns, stats = W.generate_corpus(
        ctx["graph"], ctx["host_users"], dists, n=n,
        alpha=cfg["alpha"], beta=cfg["beta"],
        credential_bonus=fo["credential_bonus"],
        min_out_degree=fo["min_foothold_out_degree"],
        max_creds=fo["max_harvest_creds"], seed=seed)
    if not campaigns:
        raise RuntimeError(f"generator produced no campaigns (seed={seed})")

    rng = np.random.default_rng(seed)
    rows, rid = [], 0
    for cid, camp in enumerate(campaigns):
        start = Wr.sample_start_time(
            ctx["hourly"], rng, max_offset=camp[-1][0],
            collection_seconds=cfg["day3"]["collection_seconds"])
        r, _ = Wr.campaign_to_rows(Wr.place_campaign(camp, start), cid, rid)
        rows.extend(r)
        rid += len(r)
    return pd.DataFrame(rows)[AUTH], stats


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
        out[k] = {"n": len(vals),
                  "real": mean_ci([c["auc_pr_real"] for c in vals]),
                  "aug": mean_ci([c["auc_pr_augmented"] for c in vals])}
    return out


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
    print("\n=== SCARCITY CURVE (mean +/- 95% CI across 5 seeds) ===")
    print(f"{'k':>3}  {'n':>2}  {'real-only':>18}  {'augmented':>18}  {'lift':>9}")
    for k in sorted(summary):
        s = summary[k]
        rm, rc = s["real"]
        am, ac = s["aug"]
        print(f"{k:>3}  {s['n']:>2}  {rm:.4f} +/- {rc:.4f}  {am:.4f} +/- {ac:.4f}  "
              f"{am - rm:>+9.4f}")


def main():
    cfg = load_config()
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


if __name__ == "__main__":
    main()
