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
