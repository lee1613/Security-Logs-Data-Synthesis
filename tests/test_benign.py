import duckdb
import pandas as pd
import pytest
import src.benign as B


def test_sample_benign_windows_and_filters_redteam_edges():
    con = duckdb.connect()
    con.execute("""CREATE TABLE auth AS SELECT * FROM (VALUES
        (100,'U1@D','U1@D','A','B','NTLM','Network','LogOn','Success'),
        (200,'U2@D','U2@D','A','C','NTLM','Network','LogOn','Success'),
        (5000,'U3@D','U3@D','A','D','NTLM','Network','LogOn','Success'),
        (150,'U9@D','U9@D','E','F','NTLM','Network','LogOn','Success')
      ) AS t(time,src_user,dst_user,src_computer,dst_computer,
             auth_type,logon_type,auth_orientation,success)""")
    redteam_edges = {("E", "F")}                    # this benign-looking row is a known attack
    df = B.sample_benign_from_relation(
        con, "auth", t_lo=0, t_hi=1000, n=10, redteam_edges=redteam_edges, seed=1)
    got = set(zip(df.src_computer, df.dst_computer))
    assert ("A", "B") in got and ("A", "C") in got   # in-window benign kept
    assert ("A", "D") not in got                      # time 5000 out of window
    assert ("E", "F") not in got                      # red-team edge filtered
    assert list(df.columns) == list(B.AUTH_COLS)
