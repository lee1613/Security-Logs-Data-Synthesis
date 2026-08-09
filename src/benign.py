"""Day-3 benign sampler: windowed random draw from auth.txt.gz, excluding
known red-team edges. Feeds Day-4 negatives (fit window -> train, holdout -> test).
"""
import pandas as pd

from src.parse import AUTH_COLS, _auth_rel, open_con


def sample_benign_from_relation(con, relation, t_lo, t_hi, n, redteam_edges, seed):
    """Random benign sample from a DuckDB relation (table name or read_csv expr)
    within [t_lo, t_hi], excluding rows on known red-team edges. AUTH_COLS order."""
    cols = ", ".join(AUTH_COLS)
    con.execute(f"SELECT setseed({(seed % 1000) / 1000.0})")
    df = con.execute(
        f"SELECT {cols} FROM {relation} "
        f"WHERE time >= {int(t_lo)} AND time <= {int(t_hi)} "
        f"USING SAMPLE reservoir({int(n)} ROWS)"
    ).df()
    if redteam_edges:
        mask = [(s, d) not in redteam_edges
                for s, d in zip(df["src_computer"], df["dst_computer"])]
        df = df[mask].reset_index(drop=True)
    return df[list(AUTH_COLS)]


def sample_benign(auth_path, t_lo, t_hi, n, redteam_edges, seed, temp_dir=None):
    """Same, driven straight off auth.txt.gz via the Day-1 read_csv relation."""
    con = open_con(temp_dir=temp_dir)
    return sample_benign_from_relation(
        con, _auth_rel(auth_path), t_lo, t_hi, n, redteam_edges, seed)
