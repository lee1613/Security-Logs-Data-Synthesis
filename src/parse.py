import os
import pickle
import tempfile

import duckdb
import numpy as np
import pandas as pd
import networkx as nx

AUTH_COLS = ("time", "src_user", "dst_user", "src_computer", "dst_computer",
             "auth_type", "logon_type", "auth_orientation", "success")


def open_con(temp_dir=None, memory_limit="4GB"):
    """DuckDB connection configured for out-of-core aggregation over the ~1B-row
    auth file: spill to a local temp dir (NOT the OneDrive tree) and cap RAM so
    DuckDB spills instead of OOMing. Tests use a plain duckdb.connect() instead."""
    if temp_dir is None:
        temp_dir = os.path.join(tempfile.gettempdir(), "duckdb_day1")
    os.makedirs(temp_dir, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{temp_dir.replace(chr(92), '/')}'")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=2")  # fewer parallel partitions => lower peak temp spill
    return con


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
    # Aggregate each orientation to its (small) distinct result FIRST, then union —
    # this avoids exploding the ~1B-row scan to ~2B rows (which overflows temp disk).
    rel = _auth_rel(auth_path)
    return con.execute(
        f"SELECT user, computer, SUM(cnt) AS cnt FROM ("
        f"  SELECT src_user AS user, src_computer AS computer, COUNT(*) AS cnt "
        f"  FROM {rel} GROUP BY 1, 2 "
        f"  UNION ALL "
        f"  SELECT src_user AS user, dst_computer AS computer, COUNT(*) AS cnt "
        f"  FROM {rel} GROUP BY 1, 2) GROUP BY user, computer"
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


def recover_redteam(con, auth_path, rt_path):
    """LEFT JOIN redteam onto auth on (time,user,src,dst); recover the 9-field
    record. Returns (full_df in AUTH_COLS order, n_unmatched)."""
    q = f"""
      SELECT r.time, r.user AS src_user, a.dst_user, r.src_computer, r.dst_computer,
             a.auth_type, a.logon_type, a.auth_orientation, a.success
      FROM {_redteam_rel(rt_path)} r
      LEFT JOIN {_auth_rel(auth_path)} a
        ON a.time = r.time AND a.src_user = r.user
       AND a.src_computer = r.src_computer AND a.dst_computer = r.dst_computer
    """
    df = con.execute(q).df()
    df = df.drop_duplicates(subset=["time", "src_user", "src_computer", "dst_computer"])
    unmatched = int(df["auth_type"].isna().sum())
    full = df[df["auth_type"].notna()].copy()
    full = full[list(AUTH_COLS)].reset_index(drop=True)
    return full, unmatched


def build_graph(edge_counts, server_in_degree_percentile=90, server_min_in_out_ratio=3.0):
    """DiGraph from edge counts. Nodes carry in/out degree, in_out_ratio,
    and a derived is_server (top in-degree percentile AND receives >> initiates)."""
    g = nx.DiGraph()
    for r in edge_counts.itertuples(index=False):
        g.add_edge(r.src_computer, r.dst_computer, weight=int(r.weight))

    for n in g.nodes:
        indeg, outdeg = g.in_degree(n), g.out_degree(n)
        g.nodes[n]["in_degree"] = int(indeg)
        g.nodes[n]["out_degree"] = int(outdeg)
        g.nodes[n]["in_out_ratio"] = indeg / (outdeg + 1)

    indeg_arr = np.array([g.nodes[n]["in_degree"] for n in g.nodes], dtype=float)
    thr = np.percentile(indeg_arr, server_in_degree_percentile)
    for n in g.nodes:
        g.nodes[n]["is_server"] = bool(
            g.nodes[n]["in_degree"] >= thr
            and g.nodes[n]["in_out_ratio"] >= server_min_in_out_ratio
        )
    return g


def save_graph(g, path):
    with open(path, "wb") as f:
        pickle.dump(g, f)


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def validate_target_attractiveness(graph, redteam_full, keep_ratio=2.0):
    """Structure-only check: are red-team target hosts high-in-degree vs all hosts?
    Returns a report dict and the keep/drop decision for the is_server proxy."""
    all_indeg = np.array([graph.nodes[n]["in_degree"] for n in graph.nodes], dtype=float)
    targets = [t for t in redteam_full["dst_computer"].unique() if t in graph]
    tgt_indeg = np.array([graph.nodes[t]["in_degree"] for t in targets], dtype=float)

    tmed = float(np.median(tgt_indeg)) if len(tgt_indeg) else 0.0
    amed = float(np.median(all_indeg)) if len(all_indeg) else 0.0
    ratio = (tmed / amed) if amed > 0 else float("inf")

    return {
        "n_targets_total": int(redteam_full["dst_computer"].nunique()),
        "n_targets_in_graph": len(targets),
        "target_indegree_median": tmed,
        "all_indegree_median": amed,
        "separation_ratio": ratio,
        "keep_server_proxy": bool(ratio >= keep_ratio),
    }
