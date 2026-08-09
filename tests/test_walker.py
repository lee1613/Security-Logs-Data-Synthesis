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
