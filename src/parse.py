import pickle

import duckdb
import numpy as np
import pandas as pd
import networkx as nx

AUTH_COLS = ("time", "src_user", "dst_user", "src_computer", "dst_computer",
             "auth_type", "logon_type", "auth_orientation", "success")


def _auth_rel(auth_path):
    # ponytail: inline the trusted config path (single-quote escaped) rather than
    # bind params — read_csv columns/compression are literals, and inlining keeps
    # one relation-builder reused by every query below.
    p = str(auth_path).replace("\\", "/").replace("'", "''")
    return (f"read_csv('{p}', header=false, delim=',', compression='gzip', "
            "columns={'time':'BIGINT','src_user':'VARCHAR','dst_user':'VARCHAR',"
            "'src_computer':'VARCHAR','dst_computer':'VARCHAR','auth_type':'VARCHAR',"
            "'logon_type':'VARCHAR','auth_orientation':'VARCHAR','success':'VARCHAR'})")


def _redteam_rel(rt_path):
    p = str(rt_path).replace("\\", "/").replace("'", "''")
    return (f"read_csv('{p}', header=false, delim=',', compression='gzip', "
            "columns={'time':'BIGINT','user':'VARCHAR','src_computer':'VARCHAR',"
            "'dst_computer':'VARCHAR'})")


def compute_edge_counts(con, auth_path):
    return con.execute(
        f"SELECT src_computer, dst_computer, COUNT(*) AS weight "
        f"FROM {_auth_rel(auth_path)} GROUP BY src_computer, dst_computer"
    ).df()


def compute_user_host_counts(con, auth_path):
    # A credential is "seen on" a host if it appears as its source OR destination.
    # UNNEST a 2-element list per row => both orientations in a single file scan.
    return con.execute(
        f"SELECT user, computer, COUNT(*) AS cnt FROM ("
        f"  SELECT src_user AS user, "
        f"         UNNEST([src_computer, dst_computer]) AS computer "
        f"  FROM {_auth_rel(auth_path)}) GROUP BY user, computer"
    ).df()


def compute_hourly_volume(con, auth_path):
    return con.execute(
        f"SELECT (time // 3600) % 24 AS hour, COUNT(*) AS cnt "
        f"FROM {_auth_rel(auth_path)} GROUP BY 1 ORDER BY 1"
    ).df()


def compute_marginals(con, auth_path):
    rel = _auth_rel(auth_path)
    out = {}
    for col in ("auth_type", "logon_type", "auth_orientation", "success"):
        out[col] = con.execute(
            f"SELECT {col} AS value, COUNT(*) AS cnt FROM {rel} GROUP BY 1"
        ).df()
    return out
