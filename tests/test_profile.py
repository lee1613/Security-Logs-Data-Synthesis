import pandas as pd
import networkx as nx
import src.profile as PR


def _rt(rows):
    return pd.DataFrame(rows, columns=[
        "time", "src_user", "dst_user", "src_computer", "dst_computer",
        "auth_type", "logon_type", "auth_orientation", "success"])


def test_sessionize_gap():
    df = _rt([
        [0, "U1@D", "U1@D", "C1", "C2", "Ntlm", "Network", "LogOn", "Success"],
        [60, "U1@D", "U1@D", "C2", "C3", "Ntlm", "Network", "LogOn", "Success"],
        [100000, "U1@D", "U1@D", "C3", "C4", "Ntlm", "Network", "LogOn", "Success"],
        [100060, "U1@D", "U1@D", "C4", "C5", "Ntlm", "Network", "LogOn", "Success"],
        [500000, "U1@D", "U1@D", "C5", "C6", "Ntlm", "Network", "LogOn", "Success"],
        [500060, "U1@D", "U1@D", "C6", "C7", "Ntlm", "Network", "LogOn", "Success"],
    ])
    out = PR.sessionize(df, gap_seconds=14400)
    assert out["campaign_id"].tolist() == [0, 0, 1, 1, 2, 2]


def test_freeze_split_whole_campaign():
    df = _rt([
        [0, "U1@D", "U1@D", "C1", "C2", "Ntlm", "Network", "LogOn", "Success"],
        [60, "U1@D", "U1@D", "C2", "C3", "Ntlm", "Network", "LogOn", "Success"],
        [100000, "U1@D", "U1@D", "C3", "C4", "Ntlm", "Network", "LogOn", "Success"],
        [100060, "U1@D", "U1@D", "C4", "C5", "Ntlm", "Network", "LogOn", "Success"],
        [500000, "U1@D", "U1@D", "C5", "C6", "Ntlm", "Network", "LogOn", "Success"],
        [500060, "U1@D", "U1@D", "C6", "C7", "Ntlm", "Network", "LogOn", "Success"],
    ])
    s = PR.sessionize(df, gap_seconds=14400)
    fit, holdout = PR.freeze_split(s, fit_fraction=0.70)
    assert sorted(fit["campaign_id"].unique().tolist()) == [0, 1]
    assert holdout["campaign_id"].unique().tolist() == [2]
    assert len(fit) == 4 and len(holdout) == 2
    # no campaign straddles the boundary
    assert set(fit["campaign_id"]).isdisjoint(set(holdout["campaign_id"]))


def test_build_profile_keys():
    fit = _rt([
        [0, "U1@D", "U1@D", "C1", "C2", "Ntlm", "Network", "LogOn", "Success"],
        [60, "U2@D", "U2@D", "C2", "C3", "Kerberos", "Network", "LogOn", "Success"],
        [120, "U1@D", "U1@D", "C3", "C4", "Ntlm", "Network", "LogOn", "Success"],
        [100000, "U1@D", "U1@D", "C1", "C3", "Ntlm", "Network", "LogOn", "Success"],
    ])
    fit = PR.sessionize(fit, gap_seconds=14400)  # -> campaigns {0:3 events, 1:1 event}
    g = nx.DiGraph()
    for u, v in [("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C1", "C3")]:
        g.add_edge(u, v)
    for n in g.nodes:
        g.nodes[n]["in_degree"] = g.in_degree(n)
        g.nodes[n]["out_degree"] = g.out_degree(n)
        g.nodes[n]["in_out_ratio"] = g.in_degree(n) / (g.out_degree(n) + 1)
        g.nodes[n]["is_server"] = g.in_degree(n) >= 2

    prof = PR.build_profile(fit, g)
    assert sorted(prof["chain_length"]["values"]) == [1, 3]
    assert prof["chain_length"]["median"] == 2.0
    assert len(prof["inter_hop_dt"]["values"]) == 2         # two positive gaps in campaign 0
    assert all(x > 0 for x in prof["inter_hop_dt"]["values"])
    for key in ("seed_hosts", "target_indegree", "conditional_attributes",
                "credential_reuse"):
        assert key in prof
    assert prof["credential_reuse"][0]["n_distinct_credentials"] == 2  # U1,U2 in campaign 0
