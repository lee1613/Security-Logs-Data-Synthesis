"""Day-4 validation statistics: SPEC V1 constraints, V2 fidelity, novelty.

This module answers "is the synthetic corpus a faithful, non-memorised stand-in
for the real red team?" — NOT "does the detector work". A failing number here is
a finding to report, never a reason to retune the generator.

It COMPUTES AND RETURNS; it does not print. Every check comes back as a
structured dict (`status` in pass/fail/n-a plus counts) so the orchestrator owns
presentation, and nothing here raises on a violated constraint — an `assert` that
kills the process is a worse bug report than a row of counts. Plotting is kept in
`plot_*` helpers so every statistic above is testable without matplotlib.

Two things the numbers do NOT prove, stated up front so they are not read as
wins:

* **Constant categoricals.** auth_type/logon_type/auth_orientation/success are
  constant `(NTLM, Network, LogOn, Success)` on BOTH sides by measurement
  (SPEC line 278), so a JS divergence of ~0 on them is a constant matching a
  constant. It is tautological and is deliberately not computed here.
* **"No credential used before acquisition"** is N/A by architecture, not a
  passing assertion — see `v1_assertions`.

Novelty has an inverted sense here. The V1 constraint forces every synthetic
edge to be a real graph edge, so a "novel" synth edge is a *real* edge the red
team did not traverse in the fit split — exactly the extra coverage synth
contributes. Read it jointly with the scarcity lift: lift + high novelty is
generalisation, lift + low novelty is memorisation.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from src.walker import fit_fanout_distributions

# The N/A item, verbatim, in one place so it cannot drift into sounding like a pass.
_CRED_NA = ("N/A by architecture: the fan-out generator has no hops -- every "
            "credential is harvested at the foothold at t=0 and used only from "
            "there, so 'used before acquisition' cannot occur by construction. "
            "Not asserted, not faked.")

_CATEGORICAL_NOTE = ("auth_type/logon_type/auth_orientation/success are constant "
                     "(NTLM, Network, LogOn, Success) on both sides by measurement, "
                     "so JS~0 there is a constant matching a constant and proves "
                     "nothing (SPEC line 278). Not computed.")


# --------------------------------------------------------------------------
# Task 7: novelty / memorisation
# --------------------------------------------------------------------------
def campaign_edge_sets(df, campaign_col="campaign_id"):
    """Campaigns as edge sets {(src_computer, dst_computer)}, ordered by id."""
    return [set(zip(g["src_computer"], g["dst_computer"]))
            for _, g in df.groupby(campaign_col, sort=True)]


def _jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def novelty_metrics(synth_campaigns, fit_campaigns, *, near_dup_threshold=0.5):
    """How much of the synthetic corpus is a rehash of the fit campaigns?

    A campaign is its SET OF EDGES on both sides — same definition, applied
    symmetrically. `max_jaccard[i]` is synth campaign i's similarity to its
    nearest fit campaign; `edge_novelty_rate` is the fraction of distinct synth
    edges no fit campaign traversed.

    ponytail: O(n_synth * 13) set intersections, ~130k on the real corpus, which
    runs in seconds. If the fit side ever grows past a few hundred campaigns,
    index edges -> campaigns and only score candidates that share one.
    """
    synth = [set(c) for c in synth_campaigns]
    fit = [set(c) for c in fit_campaigns]

    mx = np.array([max((_jaccard(s, f) for f in fit), default=0.0) for s in synth],
                  dtype=float)
    synth_edges = set().union(*synth) if synth else set()
    fit_edges = set().union(*fit) if fit else set()

    return {
        "max_jaccard": mx,
        "pct_near_duplicate": float((mx >= near_dup_threshold).mean()) if len(mx) else 0.0,
        "edge_novelty_rate": (len(synth_edges - fit_edges) / len(synth_edges)
                              if synth_edges else 0.0),
        "n_synth_edges": len(synth_edges),
        "n_fit_edges": len(fit_edges),
    }


# --------------------------------------------------------------------------
# Task 8: V1 hard constraints
# --------------------------------------------------------------------------
def _check(n_violations, detail, **extra):
    return {"status": "pass" if n_violations == 0 else "fail",
            "n_violations": int(n_violations), "detail": detail, **extra}


def v1_assertions(df, graph, *, collection_seconds, campaign_col="campaign_id"):
    """SPEC V1 hard constraints over an auth frame carrying `campaign_id`.

    Row order within a campaign IS emission order (the writer emits hop_index
    ascending), so the timestamp check reads the frame as written rather than
    re-sorting — re-sorting would launder exactly the violation it looks for.
    """
    bad_dt, min_dt = 0, None
    for _, g in df.groupby(campaign_col, sort=False):
        d = np.diff(g["time"].to_numpy())
        if len(d):
            bad_dt += int((d <= 0).sum())
            min_dt = int(d.min()) if min_dt is None else min(min_dt, int(d.min()))

    span = df.groupby(campaign_col)["time"].agg(["min", "max"])
    bad_window = int(((span["min"] < 0) | (span["max"] > collection_seconds)).sum())

    bad_edge = int(sum(not graph.has_edge(s, d) for s, d in
                       zip(df["src_computer"], df["dst_computer"])))

    checks = {
        "timestamps_strictly_increasing": _check(
            bad_dt, "intra-campaign dt > 0 in emission order", min_dt=min_dt),
        "within_collection_window": _check(
            bad_window, f"campaign times inside [0, {collection_seconds}]",
            latest_time=int(df["time"].max()) if len(df) else None),
        "edges_exist_in_graph": _check(
            bad_edge, "every (src,dst) is a real benign-graph edge"),
        # NOT a pass and NOT omitted: there is nothing here to assert.
        "no_credential_used_before_acquisition": {"status": "n/a", "detail": _CRED_NA},
    }
    checks["all_pass"] = all(c["status"] == "pass" for c in checks.values()
                             if isinstance(c, dict) and c["status"] != "n/a")
    return checks


# --------------------------------------------------------------------------
# Task 8: V2 distributional fidelity
# --------------------------------------------------------------------------
def js_divergence(a, b, *, bins=20):
    """Jensen-Shannon divergence (base 2, bits, range [0,1]) between two samples.

    ponytail: BINNING CHOICE, and JS is binning-sensitive — report the bin count
    with the number. Host in-degree is heavy-tailed (a few servers absorb
    everything), so equal-width bins on raw counts would put ~all mass in bin 0
    and report agreement that is really just saturation. We bin on log10(1+x)
    over `bins` equal-width bins spanning the union of both samples. Upgrade path
    if this ever matters more: quantile bins from the real side only.
    """
    a = np.log10(1.0 + np.asarray(a, dtype=float))
    b = np.log10(1.0 + np.asarray(b, dtype=float))
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(lo, hi if hi > lo else lo + 1.0, bins + 1)
    p = np.histogram(a, bins=edges)[0].astype(float)
    q = np.histogram(b, bins=edges)[0].astype(float)
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)


def _dst_in_degree(df, graph):
    """Per-event target in-degree; a host absent from the graph scores 0."""
    indeg = dict(graph.in_degree())
    return df["dst_computer"].map(indeg).fillna(0).to_numpy(dtype=float)


def v2_metrics(synth_df, fit_df, graph, *, bins=20):
    """KS on breadth / inter-event dt / creds-per-campaign, JS on dst in-degree.

    Per-campaign quantities come from `walker.fit_fanout_distributions` — the
    same function that fitted the generator — so the comparison cannot drift from
    the definitions the generator was built against. Both frames need a
    `campaign_id` column.

    POWER ASYMMETRY: 13 real campaigns vs thousands of synthetic ones. A KS
    p-value is not an effect size here; read `statistic` and the medians.
    """
    s = fit_fanout_distributions(synth_df)
    r = fit_fanout_distributions(fit_df)

    ks = {}
    for k in ("breadth", "inter_event_dt", "creds_per_campaign"):
        stat, p = ks_2samp(s[k], r[k])
        ks[k] = {"statistic": float(stat), "p_value": float(p),
                 "n_synth": len(s[k]), "n_real": len(r[k]),
                 "median_synth": float(np.median(s[k])),
                 "median_real": float(np.median(r[k])),
                 "mean_synth": float(np.mean(s[k])),
                 "mean_real": float(np.mean(r[k]))}

    return {
        "ks": ks,
        "dst_in_degree_js": js_divergence(_dst_in_degree(synth_df, graph),
                                          _dst_in_degree(fit_df, graph), bins=bins),
        "js_bins": bins,
        "note_constant_categoricals": _CATEGORICAL_NOTE,
        "synth": s,
        "real": r,
    }


# --------------------------------------------------------------------------
# Figures (kept apart from the statistics above on purpose)
# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")          # headless; no display, no interactive backend
    import matplotlib.pyplot as plt
    return plt


def _overlay(plt, path, synth, real, title, xlabel, log_x=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    both = np.concatenate([np.asarray(synth, float), np.asarray(real, float)])
    edges = (np.geomspace(max(both.min(), 1), both.max(), 30) if log_x
             else np.histogram_bin_edges(both, bins=25))
    for vals, label, color in ((real, f"real fit (n={len(real)})", "#c0392b"),
                               (synth, f"synth (n={len(synth)})", "#2980b9")):
        ax.hist(vals, bins=edges, density=True, alpha=0.55, label=label, color=color)
    if log_x:
        ax.set_xscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel="density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_v2_figures(v2, out_dir, graph=None, synth_df=None, fit_df=None):
    """Overlaid synth-vs-real histograms for every V2 field. Returns paths."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _plt()

    specs = [("breadth", "V2: fan-out breadth", "distinct targets per campaign", False),
             ("inter_event_dt", "V2: inter-event dt", "seconds (log)", True),
             ("creds_per_campaign", "V2: credentials per campaign",
              "distinct src_user per campaign", False)]
    paths = [_overlay(plt, out / f"v2_{k}.png", v2["synth"][k], v2["real"][k],
                      title, xlabel, log_x) for k, title, xlabel, log_x in specs]

    if graph is not None and synth_df is not None and fit_df is not None:
        paths.append(_overlay(
            plt, out / "v2_dst_in_degree.png",
            _dst_in_degree(synth_df, graph), _dst_in_degree(fit_df, graph),
            f"V2: target in-degree (JS={v2['dst_in_degree_js']:.4f})",
            "dst in-degree (log)", True))
    return paths


def plot_hourly(synth_df, hourly_df, out_dir, filename="caveat_b_hourly.png"):
    """CAVEAT B footnote, not a feature. Hour-of-day is a near-zero discriminator
    (benign peaks in the same business hours, zero off-hours attack examples), so
    it is a distributional figure only. `hour` matches parse.compute_hourly_volume."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _plt()

    s = ((synth_df["time"] // 3600) % 24).value_counts(normalize=True).sort_index()
    s = s.reindex(range(24), fill_value=0.0)
    r = hourly_df.set_index("hour")["cnt"].reindex(range(24), fill_value=0)
    r = r / r.sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(24) - 0.2, r.to_numpy(), width=0.4, label="real corpus", color="#c0392b")
    ax.bar(np.arange(24) + 0.2, s.to_numpy(), width=0.4, label="synth", color="#2980b9")
    ax.set(title="Caveat B: hour-of-day, synth vs real (figure only, never a feature)",
           xlabel="hour of day", ylabel="share of events", xticks=range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / filename, dpi=120)
    plt.close(fig)
    return str(out / filename)
