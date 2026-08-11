import hashlib

import numpy as np
import pandas as pd
import networkx as nx
import pytest
import src.walker as W


def _fit(rows):
    return pd.DataFrame(rows, columns=[
        "time", "src_user", "dst_user", "src_computer", "dst_computer",
        "auth_type", "logon_type", "auth_orientation", "success", "campaign_id"])


def test_fit_fanout_distributions():
    # campaign 0: foothold C1 -> C2,C3,C4 (breadth 3), users U1,U2 (2 creds), dts 60,60
    # campaign 1: foothold C1 -> C3 (breadth 1), user U1 (1 cred), no dt
    fit = _fit([
        [0,   "U1@D", "U1@D", "C1", "C2", "NTLM", "Network", "LogOn", "Success", 0],
        [60,  "U2@D", "U2@D", "C1", "C3", "NTLM", "Network", "LogOn", "Success", 0],
        [120, "U1@D", "U1@D", "C1", "C4", "NTLM", "Network", "LogOn", "Success", 0],
        [100000, "U1@D", "U1@D", "C1", "C3", "NTLM", "Network", "LogOn", "Success", 1],
    ])
    d = W.fit_fanout_distributions(fit)
    assert sorted(d["breadth"]) == [1, 3]
    assert sorted(d["creds_per_campaign"]) == [1, 2]
    assert sorted(d["inter_event_dt"]) == [60, 60]
    assert all(x > 0 for x in d["inter_event_dt"])


def test_build_host_users():
    uh = pd.DataFrame(
        [["U1@D", "C1", 5], ["U2@D", "C1", 2], ["U1@D", "C2", 1]],
        columns=["user", "computer", "cnt"])
    m = W.build_host_users(uh)
    assert m["C1"] == {"U1@D", "U2@D"}
    assert m["C2"] == {"U1@D"}


def _rng():
    return np.random.default_rng(42)


def test_pick_foothold_prefers_high_out_degree():
    g = nx.DiGraph()
    g.add_edge("F", "A"); g.add_edge("F", "B"); g.add_edge("F", "C")  # out 3
    g.add_edge("S", "A")                                              # out 1
    cands, cp = W.foothold_candidates(g, min_out_degree=1)
    assert set(cands) == {"F", "S"}
    # F has 3x the out-degree of S -> 0.75 vs 0.25 probability
    pF = cp[cands.index("F")]
    assert abs(pF - 0.75) < 1e-9
    pick = W.pick_foothold(cands, cp, _rng())
    assert pick in cands


def test_foothold_candidates_threshold_excludes_small():
    g = nx.DiGraph()
    g.add_edge("F", "A"); g.add_edge("F", "B")  # out 2
    g.add_edge("S", "A")                         # out 1
    cands, _ = W.foothold_candidates(g, min_out_degree=2)
    assert cands == ["F"]


def test_harvest_credentials_caps():
    host_users = {"F": {f"U{i}@D" for i in range(10)}}
    got = W.harvest_credentials("F", host_users, max_creds=4, rng=_rng())
    assert len(got) == 4
    assert got <= host_users["F"]
    # no creds seen -> empty set
    assert W.harvest_credentials("X", host_users, max_creds=4, rng=_rng()) == set()


def _graph_with_targets():
    g = nx.DiGraph()
    for dst, w in [("T1", 5), ("T2", 1), ("T3", 3)]:
        g.add_edge("F", dst, weight=w)
    # set in_degree node attrs the weighting reads
    for n in g.nodes:
        g.nodes[n]["in_degree"] = {"T1": 10, "T2": 2, "T3": 7}.get(n, 0)
    return g


def test_generate_campaign_valid_edges_and_creds():
    g = _graph_with_targets()
    host_users = {"F": {"U1@D", "U2@D"}, "T1": {"U1@D"}, "T2": {"U9@D"}, "T3": {"U2@D"}}
    dists = {"breadth": [3], "creds_per_campaign": [2], "inter_event_dt": [60, 90]}
    cands, cp = W.foothold_candidates(g, min_out_degree=1)  # only F qualifies
    events, capped = W.generate_campaign(
        g, host_users, dists, cands, cp, _rng(),
        alpha=1.0, beta=1.0, credential_bonus=2.0, max_creds=60)
    assert capped is False
    assert len(events) == 3                       # breadth 3, F has 3 targets
    for off, user, src, dst in events:
        assert src == "F"
        assert g.has_edge(src, dst)               # V1: every edge real
        assert user in {"U1@D", "U2@D"}           # only compromised creds used
        assert off > 0
    # strictly increasing offsets (Δt > 0)
    offs = [e[0] for e in events]
    assert offs == sorted(offs) and len(set(offs)) == len(offs)


def test_generate_campaign_caps_breadth_at_out_degree():
    g = _graph_with_targets()  # F has 3 targets
    host_users = {"F": {"U1@D"}, "T1": {"U1@D"}, "T2": {"U1@D"}, "T3": {"U1@D"}}
    dists = {"breadth": [10], "creds_per_campaign": [1], "inter_event_dt": [60]}
    cands, cp = W.foothold_candidates(g, min_out_degree=1)
    events, capped = W.generate_campaign(
        g, host_users, dists, cands, cp, _rng(),
        alpha=1.0, beta=1.0, credential_bonus=2.0, max_creds=60)
    assert capped is True
    assert len(events) == 3                        # capped at out-degree 3


def test_generate_campaign_discards_when_no_creds():
    g = _graph_with_targets()
    host_users = {"T1": {"U1@D"}}                  # F has no creds seen
    dists = {"breadth": [2], "creds_per_campaign": [1], "inter_event_dt": [60]}
    cands, cp = W.foothold_candidates(g, min_out_degree=1)
    events, capped = W.generate_campaign(
        g, host_users, dists, cands, cp, _rng(),
        alpha=1.0, beta=1.0, credential_bonus=2.0, max_creds=60)
    assert events is None and capped is False


def test_generate_campaign_restricts_to_sampled_cred_count():
    g = _graph_with_targets()
    # foothold F has 5 creds available, but creds_per_campaign says a campaign uses 2
    host_users = {"F": {f"U{i}@D" for i in range(5)},
                  "T1": {"U0@D"}, "T2": {"U1@D"}, "T3": {"U2@D"}}
    dists = {"breadth": [3], "creds_per_campaign": [2], "inter_event_dt": [60]}
    cands, cp = W.foothold_candidates(g, min_out_degree=1)
    events, _ = W.generate_campaign(
        g, host_users, dists, cands, cp, _rng(),
        alpha=1.0, beta=1.0, credential_bonus=2.0, max_creds=60)
    assert events is not None
    assert len({user for _, user, _, _ in events}) <= 2   # never more than sampled m


def test_generate_corpus_stats_and_validity():
    g = _graph_with_targets()
    host_users = {"F": {"U1@D", "U2@D"}, "T1": {"U1@D"}, "T2": {"U9@D"}, "T3": {"U2@D"}}
    dists = {"breadth": [2, 3], "creds_per_campaign": [2], "inter_event_dt": [60, 90]}
    campaigns, stats = W.generate_corpus(
        g, host_users, dists, n=50, alpha=1.0, beta=1.0,
        credential_bonus=2.0, min_out_degree=1, max_creds=60, seed=42)
    assert len(campaigns) == 50
    assert stats["n"] == 50
    assert 0.0 <= stats["discard_rate"] <= 1.0
    assert 0.0 <= stats["cap_rate"] <= 1.0
    # V1: every generated edge exists in the graph
    for camp in campaigns:
        for _, _, src, dst in camp:
            assert g.has_edge(src, dst)


# --- Day-4 Task 5 baseline #2: naive uniform fan-out --------------------------
#
# The naive flag exists to isolate the WEIGHTING and CREDENTIAL layers. With it
# on (and alpha=beta=0, credential_bonus=1.0 from the caller) the generator still
# picks footholds by out-degree, still samples breadth from the fit split, and
# still only emits real graph edges -- so campaign size and shape are controlled
# and the only thing that changed is how targets are weighted and how users are
# assigned. With it OFF nothing may move at all; that is what the golden below
# is for.

def _naive_graph():
    g = nx.DiGraph()
    for i, w in enumerate([5, 1, 3, 2, 7, 4]):
        g.add_edge("F", f"T{i}", weight=w)
    for n in g.nodes:
        g.nodes[n]["in_degree"] = {"T0": 10, "T1": 2, "T2": 7,
                                   "T3": 1, "T4": 30, "T5": 5}.get(n, 0)
    return g


# Captured from the generator BEFORE the naive flag existed. Hashes the
# (offset, foothold, target) projection rather than the full events: the user
# picked out of a Python set depends on str hash randomisation across processes,
# so only the structural part is reproducible enough to pin.
_PRE_NAIVE_SHA256 = "920940ce6cab9b4ef6fef63fadb2641c83335f0d9dfd9c9a1a9be7f6a009d3db"


def test_flag_off_reproduces_the_pre_naive_corpus_byte_for_byte():
    g = _naive_graph()
    # no target carries a harvested credential, so credential_bonus never fires
    # and target selection cannot depend on set iteration order
    host_users = {"F": {f"U{i}@D" for i in range(6)},
                  **{f"T{i}": {f"X{i}@D"} for i in range(6)}}
    dists = {"breadth": [2, 3, 4], "creds_per_campaign": [1, 2, 3],
             "inter_event_dt": [60, 90, 120]}
    campaigns, stats = W.generate_corpus(
        g, host_users, dists, n=40, alpha=1.0, beta=1.0, credential_bonus=2.0,
        min_out_degree=1, max_creds=60, seed=42)
    structural = repr([[(o, f, t) for o, _, f, t in c] for c in campaigns])
    assert hashlib.sha256(structural.encode()).hexdigest() == _PRE_NAIVE_SHA256
    assert stats == {"n": 40, "attempts": 40, "discard_rate": 0.0, "cap_rate": 0.0}


def test_naive_target_weights_are_uniform_and_tuned_ones_are_not():
    g = _naive_graph()
    targets = list(g.successors("F"))
    host_users = {"T0": {"U0@D"}}          # T0 would earn the credential bonus
    compromised = {"U0@D"}

    flat = W.target_weights(g, "F", targets, host_users, compromised,
                            alpha=0.0, beta=0.0, credential_bonus=1.0)
    assert np.allclose(flat, 1.0 / len(targets))

    tuned = W.target_weights(g, "F", targets, host_users, compromised,
                             alpha=1.0, beta=1.0, credential_bonus=2.0)
    assert not np.allclose(tuned, tuned[0])          # the weighting really bites
    assert abs(tuned.sum() - 1.0) < 1e-12


def _naive_corpus(naive, host_users, creds=1, seed=7):
    """Six targets and every harvestable cred available. creds=1 makes the
    creds-per-campaign restriction visible per campaign; creds=6 (>= the pool)
    disables it, isolating the user-assignment difference."""
    dists = {"breadth": [6], "creds_per_campaign": [creds], "inter_event_dt": [60]}
    kw = (dict(alpha=0.0, beta=0.0, credential_bonus=1.0) if naive
          else dict(alpha=1.0, beta=1.0, credential_bonus=2.0))
    campaigns, _ = W.generate_corpus(
        _naive_graph(), host_users, dists, n=30, min_out_degree=1,
        max_creds=60, seed=seed, naive=naive, **kw)
    return campaigns


def test_naive_skips_the_creds_per_campaign_restriction():
    # no target intersects the compromised pool, so both modes fall back to the
    # full pool for user assignment -- the ONLY difference left is the restriction
    host_users = {"F": {f"U{i}@D" for i in range(5)},
                  **{f"T{i}": {f"X{i}@D"} for i in range(6)}}
    per_campaign = lambda cs: [len({u for _, u, _, _ in c}) for c in cs]

    assert set(per_campaign(_naive_corpus(False, host_users))) == {1}   # m = 1
    assert max(per_campaign(_naive_corpus(True, host_users))) > 1       # unrestricted


def test_naive_draws_users_from_the_whole_compromised_pool():
    # every target carries exactly one of the harvested creds, so the tuned path
    # (intersect with the target's own users) is forced to that one user.
    # creds=6 >= the pool size, so the restriction is a no-op in BOTH modes and
    # user assignment is the only thing left that can differ.
    host_users = {"F": {f"U{i}@D" for i in range(6)},
                  **{f"T{i}": {f"U{i}@D"} for i in range(6)}}
    off_target = lambda cs: sum(u != f"U{t[1:]}@D" for c in cs for _, u, _, t in c)

    assert off_target(_naive_corpus(False, host_users, creds=6)) == 0   # local user
    assert off_target(_naive_corpus(True, host_users, creds=6)) > 0     # whole pool


def test_naive_corpus_still_honours_the_v1_edge_constraint():
    host_users = {"F": {f"U{i}@D" for i in range(5)},
                  **{f"T{i}": {f"X{i}@D"} for i in range(6)}}
    g = _naive_graph()
    campaigns = _naive_corpus(True, host_users)
    assert len(campaigns) == 30
    assert all(g.has_edge(s, t) for c in campaigns for _, _, s, t in c)
