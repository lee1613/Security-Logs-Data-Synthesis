import time

import duckdb

from src.config import load_config
import src.parse as P


def main():
    cfg = load_config()
    n = cfg["benchmark_rows"]
    rel = P._auth_rel(cfg["paths"]["auth_gz"])
    con = duckdb.connect()

    t = time.time()
    con.execute(
        f"SELECT src_computer, dst_computer, COUNT(*) FROM "
        f"(SELECT * FROM {rel} LIMIT {n}) GROUP BY 1, 2"
    ).fetchall()
    secs = time.time() - t

    factor = cfg["expected_total_rows"] / n
    # ~5 aggregate scans + 1 join scan over the file:
    est_hours = secs * factor * 6 / 3600
    print(f"{n:,} rows edge-grouped in {secs:.1f}s -> "
          f"full-pass estimate ~{est_hours:.2f}h")
    if est_hours > cfg["max_full_pass_hours"]:
        print("FALLBACK (SPEC §5): full pass too slow. Build the graph from a "
              "contiguous time-slice covering the fit-split window: add "
              "'WHERE time <= <cutoff>' to _auth_rel and document reduced coverage.")


if __name__ == "__main__":
    main()
