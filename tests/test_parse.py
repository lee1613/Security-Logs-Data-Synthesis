import duckdb
import pytest
import src.parse as P


@pytest.fixture
def con():
    return duckdb.connect()


def _lookup(df, keys, val_col):
    return {tuple(r[k] for k in keys): r[val_col] for _, r in df.iterrows()}


def test_edge_counts(con, mini_auth_path):
    df = P.compute_edge_counts(con, mini_auth_path)
    d = _lookup(df, ["src_computer", "dst_computer"], "weight")
    assert d[("C1", "C2")] == 2
    assert d[("C2", "C3")] == 1
    assert d[("C4", "C3")] == 1
    assert len(df) == 5


def test_user_host_counts(con, mini_auth_path):
    df = P.compute_user_host_counts(con, mini_auth_path)
    d = _lookup(df, ["user", "computer"], "cnt")
    assert d[("U1@D1", "C1")] == 2
    assert d[("U1@D1", "C2")] == 2
    assert d[("U2@D1", "C3")] == 2   # row3 dst + row5 src
    assert d[("U4@D1", "C3")] == 1


def test_hourly_volume(con, mini_auth_path):
    df = P.compute_hourly_volume(con, mini_auth_path)
    d = _lookup(df, ["hour"], "cnt")
    assert d[(0,)] == 4 and d[(1,)] == 1 and d[(2,)] == 1


def test_marginals(con, mini_auth_path):
    m = P.compute_marginals(con, mini_auth_path)
    at = _lookup(m["auth_type"], ["value"], "cnt")
    assert at[("Ntlm",)] == 3 and at[("Kerberos",)] == 2 and at[("?",)] == 1
    assert set(m.keys()) == {"auth_type", "logon_type", "auth_orientation", "success"}


def test_recover_redteam(con, mini_auth_path, mini_redteam_path):
    full, unmatched = P.recover_redteam(con, mini_auth_path, mini_redteam_path)
    assert unmatched == 1
    assert len(full) == 2
    assert list(full.columns) == list(P.AUTH_COLS)
    row = full[full["time"] == 3].iloc[0]
    assert row["auth_type"] == "Kerberos" and row["dst_computer"] == "C3"


def test_build_graph_degrees_and_is_server(con, mini_auth_path):
    edges = P.compute_edge_counts(con, mini_auth_path)
    g = P.build_graph(edges, server_in_degree_percentile=75, server_min_in_out_ratio=1.0)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 5
    assert g.nodes["C3"]["in_degree"] == 3
    assert g.nodes["C3"]["out_degree"] == 1
    assert abs(g.nodes["C3"]["in_out_ratio"] - 1.5) < 1e-9
    assert g["C1"]["C2"]["weight"] == 2
    assert g.nodes["C3"]["is_server"] is True
    assert g.nodes["C1"]["is_server"] is False


def test_validate_target_attractiveness(con, mini_auth_path, mini_redteam_path):
    edges = P.compute_edge_counts(con, mini_auth_path)
    g = P.build_graph(edges, server_in_degree_percentile=75, server_min_in_out_ratio=1.0)
    full, _ = P.recover_redteam(con, mini_auth_path, mini_redteam_path)
    res = P.validate_target_attractiveness(g, full, keep_ratio=2.0)
    assert res["n_targets_in_graph"] == 1          # C3 (C9 not in graph)
    assert res["target_indegree_median"] == 3.0
    assert res["all_indegree_median"] == 1.0
    assert res["separation_ratio"] == 3.0
    assert res["keep_server_proxy"] is True


def test_recover_redteam(con, mini_auth_path, mini_redteam_path):
    full, unmatched = P.recover_redteam(con, mini_auth_path, mini_redteam_path)
    assert unmatched == 1
    assert len(full) == 2
    assert list(full.columns) == list(P.AUTH_COLS)
    row = full[full["time"] == 3].iloc[0]
    assert row["auth_type"] == "Kerberos" and row["dst_computer"] == "C3"


def test_build_graph_degrees_and_is_server(con, mini_auth_path):
    edges = P.compute_edge_counts(con, mini_auth_path)
    g = P.build_graph(edges, server_in_degree_percentile=75, server_min_in_out_ratio=1.0)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 5
    assert g.nodes["C3"]["in_degree"] == 3
    assert g.nodes["C3"]["out_degree"] == 1
    assert abs(g.nodes["C3"]["in_out_ratio"] - 1.5) < 1e-9
    assert g["C1"]["C2"]["weight"] == 2
    assert g.nodes["C3"]["is_server"] is True
    assert g.nodes["C1"]["is_server"] is False


def test_validate_target_attractiveness(con, mini_auth_path, mini_redteam_path):
    edges = P.compute_edge_counts(con, mini_auth_path)
    g = P.build_graph(edges, server_in_degree_percentile=75, server_min_in_out_ratio=1.0)
    full, _ = P.recover_redteam(con, mini_auth_path, mini_redteam_path)
    res = P.validate_target_attractiveness(g, full, keep_ratio=2.0)
    assert res["n_targets_in_graph"] == 1          # C3 (C9 not in graph)
    assert res["target_indegree_median"] == 3.0
    assert res["all_indegree_median"] == 1.0
    assert res["separation_ratio"] == 3.0
    assert res["keep_server_proxy"] is True
