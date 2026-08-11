import numpy as np
import pandas as pd
import networkx as nx
import pytest

import src.validate as V
from src.parse import AUTH_COLS

WINDOW = 5011200


# --------------------------------------------------------------------------
# fixtures: tiny hand-built campaigns
# --------------------------------------------------------------------------
def _camp(cid, events):
    """events = [(time, user, src, dst)] -> AUTH_COLS frame + campaign_id."""
    return pd.DataFrame(
        [[t, u, u, s, d, "NTLM", "Network", "LogOn", "Success", cid]
         for t, u, s, d in events],
        columns=list(AUTH_COLS) + ["campaign_id"])


def _graph():
    """A->B, A->C, A->D, B->C. (C,A) and (D,A) are deliberately NOT edges."""
    g = nx.DiGraph()
    for s, d in [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C")]:
        g.add_edge(s, d, weight=1)
    for n in g.nodes:
        g.nodes[n]["in_degree"] = int(g.in_degree(n))
    return g


def _clean():
    return pd.concat([
        _camp(0, [(100, "U1@D", "A", "B"), (200, "U1@D", "A", "C")]),
        _camp(1, [(300, "U2@D", "A", "D"), (450, "U2@D", "B", "C")]),
    ], ignore_index=True)


def _corpus(n, breadth, dt, n_creds=1, start=1000, cid0=0):
    """n campaigns, each `breadth` events on distinct targets, spaced `dt`."""
    parts = []
    for c in range(n):
        parts.append(_camp(cid0 + c, [
            (start + (i + 1) * dt, f"U{i % n_creds}@D", "A", f"T{i}")
            for i in range(breadth)]))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Task 7 -- novelty_metrics
# --------------------------------------------------------------------------
def test_identical_edge_sets_are_jaccard_one_and_zero_novelty():
    c = [("A", "B"), ("B", "C")]
    out = V.novelty_metrics([c], [c])
    assert out["max_jaccard"].tolist() == [1.0]
    assert out["pct_near_duplicate"] == 1.0
    assert out["edge_novelty_rate"] == 0.0


def test_disjoint_edge_sets_are_jaccard_zero_and_full_novelty():
    out = V.novelty_metrics([[("A", "B")]], [[("X", "Y")]])
    assert out["max_jaccard"].tolist() == [0.0]
    assert out["pct_near_duplicate"] == 0.0
    assert out["edge_novelty_rate"] == 1.0


def test_partial_overlap_is_the_exact_jaccard_fraction():
    # {AB,AC} vs {AB,AD}: intersection 1, union 3
    out = V.novelty_metrics([[("A", "B"), ("A", "C")]],
                            [[("A", "B"), ("A", "D")]])
    assert out["max_jaccard"][0] == pytest.approx(1 / 3)
    # 2 distinct synth edges, AB was traversed in fit -> 1/2 novel
    assert out["edge_novelty_rate"] == pytest.approx(0.5)


def test_max_jaccard_takes_the_best_fit_campaign():
    synth = [[("A", "B"), ("A", "C")]]
    fit = [[("X", "Y")],                          # 0.0
           [("A", "B"), ("A", "D")],              # 1/3
           [("A", "B"), ("A", "C")]]              # 1.0
    assert V.novelty_metrics(synth, fit)["max_jaccard"][0] == pytest.approx(1.0)


def test_near_duplicate_threshold_is_inclusive_at_the_boundary():
    # {AB,AC} vs {AB,AC,AD,AE}: intersection 2, union 4 -> exactly 0.5
    synth = [[("A", "B"), ("A", "C")]]
    fit = [[("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")]]
    assert V.novelty_metrics(synth, fit)["max_jaccard"][0] == pytest.approx(0.5)
    assert V.novelty_metrics(synth, fit, near_dup_threshold=0.5)[
        "pct_near_duplicate"] == 1.0
    assert V.novelty_metrics(synth, fit, near_dup_threshold=0.51)[
        "pct_near_duplicate"] == 0.0


def test_edge_novelty_rate_is_over_distinct_synth_edges():
    # distinct synth edges {AB, AC, AD}; fit traversed only AB -> 2/3 novel
    synth = [[("A", "B"), ("A", "C")], [("A", "C"), ("A", "D")]]
    out = V.novelty_metrics(synth, [[("A", "B")]])
    assert out["edge_novelty_rate"] == pytest.approx(2 / 3)
    assert len(out["max_jaccard"]) == 2


def test_campaign_edge_sets_groups_by_campaign_id():
    sets = V.campaign_edge_sets(_clean())
    assert sets == [{("A", "B"), ("A", "C")}, {("A", "D"), ("B", "C")}]


# --------------------------------------------------------------------------
# Task 8 -- V1 assertions
# --------------------------------------------------------------------------
def test_v1_passes_on_clean_input():
    out = V.v1_assertions(_clean(), _graph(), collection_seconds=WINDOW)
    assert out["timestamps_strictly_increasing"]["status"] == "pass"
    assert out["within_collection_window"]["status"] == "pass"
    assert out["edges_exist_in_graph"]["status"] == "pass"


def test_v1_catches_non_increasing_timestamp():
    df = _clean()
    df.loc[1, "time"] = 50                  # earlier than the event before it
    out = V.v1_assertions(df, _graph(), collection_seconds=WINDOW)
    assert out["timestamps_strictly_increasing"]["status"] == "fail"
    assert out["timestamps_strictly_increasing"]["n_violations"] == 1


def test_v1_catches_zero_delta():
    df = _clean()
    df.loc[1, "time"] = 100                 # equal to the event before it
    out = V.v1_assertions(df, _graph(), collection_seconds=WINDOW)
    assert out["timestamps_strictly_increasing"]["status"] == "fail"
    assert out["timestamps_strictly_increasing"]["min_dt"] == 0


def test_v1_catches_campaign_past_the_window():
    df = _clean()
    df.loc[3, "time"] = WINDOW + 1
    out = V.v1_assertions(df, _graph(), collection_seconds=WINDOW)
    assert out["within_collection_window"]["status"] == "fail"
    assert out["within_collection_window"]["n_violations"] == 1


def test_v1_catches_edge_absent_from_the_graph():
    df = _clean()
    df.loc[0, "dst_computer"] = "Z"         # A->Z is not a real edge
    out = V.v1_assertions(df, _graph(), collection_seconds=WINDOW)
    assert out["edges_exist_in_graph"]["status"] == "fail"
    assert out["edges_exist_in_graph"]["n_violations"] == 1


def test_v1_credential_acquisition_is_reported_as_na_not_as_a_pass():
    out = V.v1_assertions(_clean(), _graph(), collection_seconds=WINDOW)
    item = out["no_credential_used_before_acquisition"]
    assert item["status"] == "n/a"
    assert item["status"] != "pass"
    assert "foothold" in item["detail"]     # the one-line architectural reason


def test_v1_all_pass_flag_ignores_the_na_item():
    clean = V.v1_assertions(_clean(), _graph(), collection_seconds=WINDOW)
    assert clean["all_pass"] is True
    df = _clean()
    df.loc[1, "time"] = 50
    assert V.v1_assertions(df, _graph(), collection_seconds=WINDOW)["all_pass"] is False


# --------------------------------------------------------------------------
# Task 8 -- V2 distributional statistics
# --------------------------------------------------------------------------
def test_js_divergence_is_zero_on_identical_samples():
    x = np.array([1, 2, 2, 5, 9, 40, 100])
    assert V.js_divergence(x, x) == pytest.approx(0.0, abs=1e-12)


def test_js_divergence_is_large_on_clearly_different_samples():
    assert V.js_divergence(np.ones(500), np.full(500, 10000.0)) > 0.9


def test_v2_identical_corpora_give_no_evidence_of_difference():
    df = _corpus(20, breadth=3, dt=10)
    out = V.v2_metrics(df, df.copy(), _graph())
    for k in ("breadth", "inter_event_dt", "creds_per_campaign"):
        assert out["ks"][k]["statistic"] == pytest.approx(0.0)
        assert out["ks"][k]["p_value"] > 0.99
    assert out["dst_in_degree_js"] == pytest.approx(0.0, abs=1e-12)


def test_v2_rejects_on_clearly_different_corpora():
    synth = _corpus(60, breadth=6, dt=100, n_creds=3)
    real = _corpus(13, breadth=2, dt=5, n_creds=1)
    out = V.v2_metrics(synth, real, _graph())
    for k in ("breadth", "inter_event_dt", "creds_per_campaign"):
        assert out["ks"][k]["p_value"] < 0.05, k
        assert out["ks"][k]["statistic"] > 0.9, k


def test_v2_quantities_match_the_walker_fit_definitions():
    from src.walker import fit_fanout_distributions
    df = _corpus(5, breadth=4, dt=7, n_creds=2)
    out = V.v2_metrics(df, df.copy(), _graph())
    assert out["synth"] == fit_fanout_distributions(df)
    assert out["synth"]["breadth"] == [4] * 5
    assert out["synth"]["creds_per_campaign"] == [2] * 5
    assert set(out["synth"]["inter_event_dt"]) == {7}


def test_v2_carries_the_constant_categorical_caveat():
    df = _corpus(5, breadth=3, dt=10)
    note = V.v2_metrics(df, df.copy(), _graph())["note_constant_categoricals"]
    assert "constant" in note.lower()
