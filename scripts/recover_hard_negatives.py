"""Recover the hard negatives the Day-3 benign sampler deleted.

WHY THIS EXISTS
src/benign.py drops every sampled row whose (src_computer, dst_computer) is a
red-team edge, using the edge set of ALL 701 red-team events. That looked like
label hygiene, but it makes the positive and negative classes EDGE-DISJOINT by
construction: measured on the Day-3 outputs, the 51 holdout positives and the
204,228 holdout negatives share exactly 0 edges. The region of feature space
occupied by red-team edges is therefore guaranteed empty of negatives in both
the train and the eval split, so a detector learns "this region => positive" on
fit and is still right on holdout -- not by generalizing, but because the same
global filter carved the same hole in both. Measured effect: edge_rarity alone
ranks at AUC-PR 0.550, but the trained GBT hits 0.999.

Only ~8,790 benign events sit on those 302 edges corpus-wide. They are few, and
they are exactly the hard negatives -- the benign traffic that looks structurally
like the attack. This script scans auth.txt.gz once, pulls them back, and writes
them as separate files so the originals stay auditable and the merge is
idempotent. Day-4 negatives = benign_{fit,holdout}.csv + hard_neg_{fit,holdout}.csv.

The 701 real red-team events are excluded by an anti-join on the full
(time, src_user, src_computer, dst_computer) key -- those are positives, not
negatives, and must never leak into the negative class.

Run: python scripts/recover_hard_negatives.py   (one full pass over the ~1B-row gz)
"""
import time

import pandas as pd

from src.config import load_config
import src.parse as P


def main():
    cfg = load_config()
    p, d3 = cfg["paths"], cfg["day3"]

    rt = pd.read_csv(p["redteam_full"])
    edges = rt[["src_computer", "dst_computer"]].drop_duplicates().reset_index(drop=True)
    events = rt[["time", "src_user", "src_computer", "dst_computer"]].drop_duplicates()
    fit_lo, fit_hi = d3["fit_window"]
    hold_lo, hold_hi = d3["holdout_window"]

    print(f"red-team edges: {len(edges):,}   red-team events: {len(events):,}")
    print(f"fit window [{fit_lo:,}, {fit_hi:,}]   holdout window [{hold_lo:,}, {hold_hi:,}]")

    con = P.open_con()
    con.register("rt_edges", edges)
    con.register("rt_events", events)

    cols = ", ".join(f"a.{c}" for c in P.AUTH_COLS)
    # One pass: keep only rows on a red-team edge inside either window, then
    # drop the red-team events themselves via a NOT EXISTS anti-join. Everything
    # surviving is benign traffic on an attacker-used edge -- the hard negatives.
    q = f"""
      SELECT {cols}
      FROM {P._auth_rel(p['auth_gz'])} a
      JOIN rt_edges e
        ON a.src_computer = e.src_computer AND a.dst_computer = e.dst_computer
      WHERE ((a.time BETWEEN {int(fit_lo)} AND {int(fit_hi)})
          OR (a.time BETWEEN {int(hold_lo)} AND {int(hold_hi)}))
        AND NOT EXISTS (
              SELECT 1 FROM rt_events r
              WHERE r.time = a.time AND r.src_user = a.src_user
                AND r.src_computer = a.src_computer
                AND r.dst_computer = a.dst_computer)
    """
    t0 = time.time()
    df = con.execute(q).df()
    print(f"scan finished in {time.time() - t0:.0f}s -- {len(df):,} hard negatives recovered")

    df = df[list(P.AUTH_COLS)]
    fit = df[df["time"].between(fit_lo, fit_hi)].reset_index(drop=True)
    hold = df[df["time"].between(hold_lo, hold_hi)].reset_index(drop=True)
    fit.to_csv("data/derived/hard_neg_fit.csv", index=False)
    hold.to_csv("data/derived/hard_neg_holdout.csv", index=False)

    # sanity: none of these may be an actual red-team event
    rt_keys = set(zip(events["time"], events["src_user"],
                      events["src_computer"], events["dst_computer"]))
    overlap = sum(1 for k in zip(df["time"], df["src_user"],
                                 df["src_computer"], df["dst_computer"]) if k in rt_keys)

    print(f"\nhard_neg_fit.csv      {len(fit):,} rows")
    print(f"hard_neg_holdout.csv  {len(hold):,} rows")
    print(f"[{'x' if overlap == 0 else ' '}] no red-team event leaked into the negatives (overlap={overlap})")
    print(f"[i] new holdout negative count will be {204228 + len(hold):,} "
          f"(was 204,228); base rate 51/{204228 + len(hold):,}")


if __name__ == "__main__":
    main()
