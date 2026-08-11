import numpy as np
import pandas as pd
import networkx as nx
import pytest

import src.features as F
from src.parse import AUTH_COLS


def _rows(rows):
    return pd.DataFrame(rows, columns=list(AUTH_COLS))


def _tiny_graph():
    """A->B w=3, A->C w=1, B->C w=6.  total_weight = 10.
    degrees: A in0/out2, B in1/out1, C in2/out0."""
    g = nx.DiGraph()
    g.add_edge("A", "B", weight=3)
    g.add_edge("A", "C", weight=1)
    g.add_edge("B", "C", weight=6)
    return g


# U1 seen on A and B (2 hosts), U2 seen on C only (1 host).
_HOST_USERS = {"A": {"U1@D"}, "B": {"U1@D"}, "C": {"U2@D"}}


def _row(t, user, src, dst, auth="NTLM", logon="Network"):
    return [t, user, user, src, dst, auth, logon, "LogOn", "Success"]


def test_known_values_on_tiny_graph():
    df = _rows([
        _row(1, "U1@D", "A", "B"),   # seen edge w=3
        _row(2, "U2@D", "B", "C"),   # seen edge w=6
        _row(3, "U1@D", "A", "C"),   # seen edge w=1 (rarest)
    ])
    out = F.build_features(df, _tiny_graph(), _HOST_USERS, groups=("structural",))

    assert out["edge_rarity"].tolist() == pytest.approx(
        [np.log(10 / 3), np.log(10 / 6), np.log(10 / 1)])
    assert out["dst_in_degree"].tolist() == [1, 2, 2]
    assert out["src_out_degree"].tolist() == [2, 1, 2]
    assert out["n_hosts_for_cred"].tolist() == [2, 1, 2]


def test_unseen_edge_gets_max_rarity():
    df = _rows([
        _row(1, "U1@D", "A", "C"),   # rarest observed edge (w=1)
        _row(2, "U1@D", "C", "A"),   # edge absent, both nodes present
        _row(3, "U9@D", "Z", "Q"),   # both nodes absent from the graph entirely
    ])
    out = F.build_features(df, _tiny_graph(), _HOST_USERS, groups=("structural",))
    r = out["edge_rarity"].to_numpy()

    assert np.isfinite(r).all()
    assert (r[1:] > r[0]).all()          # strictly above the rarest observed edge
    assert r[1] == r[2]                  # unseen is unseen, however it is unseen
    # missing nodes degrade to zero degree, not a crash
    assert out.loc[2, "dst_in_degree"] == 0
    assert out.loc[2, "src_out_degree"] == 0
    assert out.loc[2, "n_hosts_for_cred"] == 0


def test_leakage_guard_holdout_rows_do_not_change_fit_rows():
    fit = _rows([
        _row(1, "U1@D", "A", "B"),
        _row(2, "U2@D", "B", "C"),
        _row(3, "U1@D", "A", "C"),
    ])
    # holdout rows chosen to be maximally tempting to any fitted statistic:
    # new users, new hosts, new categories, wildly different scale
    holdout = _rows([
        _row(9999, "U7@D", "Z", "Q", auth="Kerberos", logon="Interactive"),
        _row(9999, "U8@D", "Q", "Z", auth="Wave", logon="Batch"),
    ] * 50)

    alone = F.build_features(fit, _tiny_graph(), _HOST_USERS)
    together = F.build_features(pd.concat([fit, holdout], ignore_index=True),
                                _tiny_graph(), _HOST_USERS)

    assert list(alone.columns) == list(together.columns)
    pd.testing.assert_frame_equal(alone, together.iloc[:len(fit)])


def test_deterministic():
    df = _rows([_row(1, "U1@D", "A", "B"), _row(2, "U2@D", "B", "C")])
    a = F.build_features(df, _tiny_graph(), _HOST_USERS)
    b = F.build_features(df, _tiny_graph(), _HOST_USERS)
    pd.testing.assert_frame_equal(a, b)


def test_groups_structural_only_is_a_real_ablation():
    df = _rows([_row(1, "U1@D", "A", "B")])
    g, hu = _tiny_graph(), _HOST_USERS
    struct = F.build_features(df, g, hu, groups=("structural",))
    both = F.build_features(df, g, hu)

    assert list(struct.columns) == list(F.STRUCTURAL_COLS)
    assert not any(c.startswith(("auth_type_", "logon_type_")) for c in struct.columns)
    assert list(both.columns) == list(F.STRUCTURAL_COLS) + list(F.ATTRIBUTE_COLS)
    # attribute-only is also a valid slice
    attr = F.build_features(df, g, hu, groups=("attribute",))
    assert list(attr.columns) == list(F.ATTRIBUTE_COLS)


def test_structural_only_equals_a_column_slice_of_the_full_matrix():
    """Task 6's ablation slices the structural columns out of the already-built
    full matrix instead of rebuilding 486k negatives a second time. That is only
    legitimate because build_features is a pure row-local map with a fixed emit
    order -- assert it, so the shortcut cannot rot."""
    df = _rows([_row(1, "U1@D", "A", "B"), _row(2, "U2@D", "B", "C"),
                _row(3, "U9@D", "Z", "Q")])
    g, hu = _tiny_graph(), _HOST_USERS
    pd.testing.assert_frame_equal(
        F.build_features(df, g, hu, groups=("structural",)),
        F.build_features(df, g, hu)[list(F.STRUCTURAL_COLS)])


def test_unknown_group_raises():
    df = _rows([_row(1, "U1@D", "A", "B")])
    with pytest.raises(ValueError):
        F.build_features(df, _tiny_graph(), _HOST_USERS, groups=("structural", "hour"))


def test_empty_group_raises():
    # an empty tuple has no unknown members, so it must be rejected on its own
    df = _rows([_row(1, "U1@D", "A", "B")])
    with pytest.raises(ValueError, match="must not be empty"):
        F.build_features(df, _tiny_graph(), _HOST_USERS, groups=())


def test_attribute_onehot_is_fixed_width_and_buckets_unseen():
    g, hu = _tiny_graph(), _HOST_USERS
    a = F.build_features(_rows([_row(1, "U1@D", "A", "B", "NTLM", "Network")]),
                         g, hu, groups=("attribute",))
    b = F.build_features(_rows([_row(1, "U1@D", "A", "B", "Kerberos", "Service")]),
                         g, hu, groups=("attribute",))
    c = F.build_features(_rows([_row(1, "U1@D", "A", "B", "Wave", "NotAThing")]),
                         g, hu, groups=("attribute",))

    assert list(a.columns) == list(b.columns) == list(c.columns)
    assert a.loc[0, "auth_type_NTLM"] == 1 and a["auth_type_other"].iloc[0] == 0
    assert b.loc[0, "logon_type_Service"] == 1
    # out-of-vocabulary values land in _other, exactly once each
    assert c.loc[0, "auth_type_other"] == 1 and c.loc[0, "logon_type_other"] == 1
    at = [x for x in a.columns if x.startswith("auth_type_")]
    assert a[at].sum(axis=1).tolist() == [1]     # one-hot, not multi-hot


def test_excluded_features_stay_excluded():
    df = _rows([_row(1, "U1@D", "A", "B")])
    cols = F.build_features(df, _tiny_graph(), _HOST_USERS).columns
    for banned in ("hour", "success", "orientation", "time"):
        assert not any(banned in c.lower() for c in cols)


def test_credential_novelty_is_not_emitted():
    # Regression guard. The Day-4 plan specified credential_novelty; it measured
    # 0.0000 on every real row (650 redteam_fit, 200k benign_fit) and 0.3664 on
    # synth, making it a "this row is synthetic" marker. Read the module docstring
    # of src/features.py before re-adding it.
    df = _rows([_row(1, "U1@D", "A", "C")])   # U1 never seen on C
    cols = F.build_features(df, _tiny_graph(), _HOST_USERS).columns
    assert "credential_novelty" not in cols
    assert "credential_novelty" not in F.STRUCTURAL_COLS


def test_index_aligned_and_numeric():
    df = _rows([_row(1, "U1@D", "A", "B"), _row(2, "U2@D", "B", "C")])
    df.index = [17, 42]
    out = F.build_features(df, _tiny_graph(), _HOST_USERS)
    assert out.index.tolist() == [17, 42]
    assert len(out) == len(df)
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes)
    assert not out.isna().any().any()


def test_empty_frame_keeps_schema():
    out = F.build_features(_rows([]), _tiny_graph(), _HOST_USERS)
    assert len(out) == 0
    assert list(out.columns) == list(F.STRUCTURAL_COLS) + list(F.ATTRIBUTE_COLS)
