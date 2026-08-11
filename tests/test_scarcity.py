"""Day-4 Task 4 (scarcity curve) mechanics.

These tests exist to guard the headline claim, which is a leakage claim: at
scarcity level k the generator must be refit on ONLY the k drawn campaigns. The
pre-existing 10k-campaign synth corpus was fit on all 13 fit campaigns, so
reusing it at k=1 would smuggle the other 12 campaigns' structure into the
scarce arm and inflate the lift exactly where the claim lives.

The refit test asserts on the returned DISTRIBUTIONS, not on generated output:
the distributions are where the restriction is either honoured or lost, and an
assertion on downstream samples would pass for the wrong reasons.
"""
import numpy as np
import pandas as pd
import networkx as nx
import pytest

import scripts.run_day4 as D
import src.parse as P
import src.walker as W

# hand-computed ground truth for the fixture below: campaign -> (breadth, creds)
BREADTH = {0: 3, 1: 1, 2: 2, 3: 4, 4: 1}
CREDS = {0: 2, 1: 1, 2: 2, 3: 1, 4: 1}
_USERS = {0: 2, 1: 1, 2: 2, 3: 1, 4: 1}


def _fit_df():
    """Five campaigns whose breadth/creds are known by construction (BREADTH/CREDS)."""
    rows = []
    for cid, breadth in BREADTH.items():
        users = [f"U{cid}_{j}@D" for j in range(_USERS[cid])]
        for i in range(breadth):
            u = users[i % len(users)]
            rows.append([1000 * cid + 60 * i, u, u, f"F{cid}", f"T{cid}_{i}",
                         "NTLM", "Network", "LogOn", "Success", cid])
    return pd.DataFrame(rows, columns=list(P.AUTH_COLS) + ["campaign_id"])


def _tiny_graph():
    g = nx.DiGraph()
    for i, w in enumerate([5, 1, 3, 2]):
        g.add_edge("F", f"T{i}", weight=w)
    for n in g.nodes:
        g.nodes[n]["in_degree"] = 3
    host_users = {"F": {"U1@D", "U2@D"}, "T0": {"U1@D"}, "T1": {"U2@D"},
                  "T2": {"U1@D"}, "T3": {"U2@D"}}
    return g, host_users


# --- the headline guard -------------------------------------------------------

def test_refit_at_k_uses_only_the_k_drawn_campaigns():
    fit = _fit_df()
    ids = D.draw_campaigns(fit["campaign_id"], k=2, seed=7)
    sub = D.subset_campaigns(fit, ids)

    assert len(ids) == 2
    assert set(sub["campaign_id"]) == set(ids)      # no other campaign came along
    assert len(sub) == sum(BREADTH[i] for i in ids)  # and no extra rows

    d = W.fit_fanout_distributions(sub)
    # one entry per drawn campaign, and the entries are exactly those campaigns'
    assert len(d["breadth"]) == 2
    assert sorted(d["breadth"]) == sorted(BREADTH[i] for i in ids)
    assert len(d["creds_per_campaign"]) == 2
    assert sorted(d["creds_per_campaign"]) == sorted(CREDS[i] for i in ids)
    # nothing from an undrawn campaign leaked in
    assert not (set(d["breadth"]) - {BREADTH[i] for i in ids})

    d_all = W.fit_fanout_distributions(fit)         # the all-13 analogue
    assert len(d_all["breadth"]) == 5               # the refit is a real restriction


def test_refit_at_every_k_has_exactly_k_entries():
    fit = _fit_df()
    for k in (1, 2, 3, 4, 5):
        ids = D.draw_campaigns(fit["campaign_id"], k=k, seed=k)
        d = W.fit_fanout_distributions(D.subset_campaigns(fit, ids))
        assert len(d["breadth"]) == k
        assert len(d["creds_per_campaign"]) == k


def test_k1_refit_degenerates_to_one_value_and_generation_still_succeeds():
    fit = _fit_df()
    ids = D.draw_campaigns(fit["campaign_id"], k=1, seed=3)
    d = W.fit_fanout_distributions(D.subset_campaigns(fit, ids))
    # degenerate by design: with one campaign to learn from there is one breadth
    assert len(d["breadth"]) == 1 and len(d["creds_per_campaign"]) == 1
    assert d["inter_event_dt"]                       # never empty

    g, host_users = _tiny_graph()
    campaigns, stats = W.generate_corpus(
        g, host_users, d, n=20, alpha=1.0, beta=1.0, credential_bonus=2.0,
        min_out_degree=1, max_creds=10, seed=1)
    assert len(campaigns) == 20
    assert all(g.has_edge(s, t) for c in campaigns for _, _, s, t in c)


# --- reproducibility ----------------------------------------------------------

def test_draw_is_seed_reproducible_and_seed_sensitive():
    ids = list(range(13))
    assert D.draw_campaigns(ids, 4, 42) == D.draw_campaigns(ids, 4, 42)
    assert D.draw_campaigns(ids, 4, 42) == D.draw_campaigns(np.array(ids), 4, 42)
    # a different seed must actually move the draw, or the test proves nothing
    draws = {tuple(D.draw_campaigns(ids, 4, s)) for s in range(20)}
    assert len(draws) > 1


def test_draw_rejects_k_larger_than_the_pool():
    with pytest.raises(ValueError):
        D.draw_campaigns([0, 1, 2], 4, 42)


# --- training-set assembly ----------------------------------------------------

def test_stack_labels_stay_aligned_with_rows():
    pos_a = pd.DataFrame({"f": [1.0, 2.0]})
    pos_b = pd.DataFrame({"f": [3.0]})
    neg = pd.DataFrame({"f": [4.0, 5.0, 6.0]})
    X, y = D.stack([pos_a, pos_b], neg)
    assert len(X) == len(y) == 6
    assert list(y) == [1, 1, 1, 0, 0, 0]
    assert list(X["f"]) == [1, 2, 3, 4, 5, 6]


def test_stack_rejects_frames_whose_columns_disagree():
    """Task 6 slices the negatives down to the structural columns. If a caller
    forgets to slice the positives too, pd.concat unions the columns and pads
    with NaN -- a silently mis-shaped training matrix, not an error."""
    pos = pd.DataFrame({"a": [1.0], "b": [2.0]})
    neg = pd.DataFrame({"a": [3.0]})
    with pytest.raises(ValueError):
        D.stack([pos], neg)


# --- the real splits ----------------------------------------------------------

def test_load_splits_keeps_the_holdout_out_of_training():
    sp = D.load_splits()
    cfg = D.load_config()

    # campaign level: the 13 fit campaigns and the 6 holdout campaigns are disjoint
    assert set(sp["train_pos"]["campaign_id"]) == set(range(13))
    assert set(sp["eval_pos"]["campaign_id"]) == set(range(13, 19))
    assert not set(sp["train_pos"]["campaign_id"]) & set(sp["eval_pos"]["campaign_id"])

    # row level: fit and holdout live in disjoint time windows, so no training row
    # (positive OR negative) can be a holdout row
    fit_hi = cfg["day3"]["fit_window"][1]
    hold_lo = cfg["day3"]["holdout_window"][0]
    assert fit_hi < hold_lo
    for name in ("train_pos", "train_neg"):
        assert sp[name]["time"].max() <= fit_hi, name
    for name in ("eval_pos", "eval_neg"):
        assert sp[name]["time"].min() >= hold_lo, name


def test_load_splits_merges_the_recovered_hard_negatives():
    """The leakage repair, stated as a property rather than a row count: after the
    merge the classes are no longer edge-disjoint on either side of the split."""
    sp = D.load_splits()
    edges = lambda df: set(zip(df["src_computer"], df["dst_computer"]))
    assert edges(sp["train_neg"]) & edges(sp["train_pos"])
    assert edges(sp["eval_neg"]) & edges(sp["eval_pos"])


# --- Task 9: holdout base-rate down-sampler -----------------------------------
#
# The sweep moves the TEST-TIME base rate only. All 51 real holdout positives
# must survive every point -- dropping one would change what is being measured
# from "how does precision behave as the haystack shrinks" to "how does it behave
# on a different, smaller set of attacks".

_N_POS, _N_NEG = 51, 60_000


def _eval_y():
    return np.concatenate([np.ones(_N_POS), np.zeros(_N_NEG)])


@pytest.mark.parametrize("rate", [0.01, 0.001, None])
def test_downsample_keeps_every_positive(rate):
    y = _eval_y()
    idx = D.downsample_eval(y, rate, seed=42)
    assert (y[idx] == 1).sum() == _N_POS
    assert set(np.flatnonzero(y == 1)) <= set(idx)


@pytest.mark.parametrize("rate", [0.01, 0.001])
def test_downsample_hits_the_target_base_rate(rate):
    y = _eval_y()
    achieved = y[D.downsample_eval(y, rate, seed=42)].mean()
    assert abs(achieved - rate) < rate * 0.01      # rounding on 51 positives only


def test_downsample_none_returns_everything_untouched():
    y = _eval_y()
    idx = D.downsample_eval(y, None, seed=42)
    assert np.array_equal(idx, np.arange(len(y)))


def test_downsample_is_seed_reproducible_and_seed_sensitive():
    y = _eval_y()
    assert np.array_equal(D.downsample_eval(y, 0.01, seed=42),
                          D.downsample_eval(y, 0.01, seed=42))
    assert not np.array_equal(D.downsample_eval(y, 0.01, seed=42),
                              D.downsample_eval(y, 0.01, seed=43))


def test_downsample_rejects_a_rate_below_the_natural_floor():
    # 51 positives against only 60k negatives cannot reach 1:1e6
    with pytest.raises(ValueError):
        D.downsample_eval(_eval_y(), 1e-6, seed=42)


def test_load_splits_returns_auth_cols_in_order():
    sp = D.load_splits()
    for name in ("train_neg", "eval_neg"):
        assert list(sp[name].columns) == list(P.AUTH_COLS), name
    for name in ("train_pos", "eval_pos"):
        assert list(sp[name].columns) == list(P.AUTH_COLS) + ["campaign_id"], name
