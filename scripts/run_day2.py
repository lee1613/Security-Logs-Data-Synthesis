import pickle
import time

import numpy as np
import pandas as pd

from src.config import load_config
import src.parse as P
import src.walker as W


def main():
    cfg = load_config()
    p = cfg["paths"]
    fo = cfg["fanout"]

    graph = P.load_graph(p["graph"])
    with open(p["aggregates"], "rb") as f:
        agg = pickle.load(f)
    fit = pd.read_csv(p["redteam_fit"])

    dists = W.fit_fanout_distributions(fit)
    host_users = W.build_host_users(agg["user_host"])

    n = fo["n_campaigns"]
    t = time.time()
    campaigns, stats = W.generate_corpus(
        graph, host_users, dists, n=n,
        alpha=cfg["alpha"], beta=cfg["beta"],
        credential_bonus=fo["credential_bonus"],
        min_out_degree=fo["min_foothold_out_degree"],
        max_creds=fo["max_harvest_creds"], seed=cfg["seed"])
    secs = time.time() - t

    out_path = p["graph"].replace("graph.pkl", "synth_campaigns.pkl")
    W.save_campaigns(campaigns, out_path)

    # V1 (subset): every generated edge exists in the graph
    bad = sum(1 for camp in campaigns for _, _, s, d in camp if not graph.has_edge(s, d))

    syn_breadth = [len(c) for c in campaigns]
    real_breadth = dists["breadth"]

    print(f"generated {len(campaigns):,} campaigns in {secs:.1f}s "
          f"({len(campaigns)/max(secs,1e-9):,.0f}/s)")
    print(f"discard_rate={stats['discard_rate']:.3f} cap_rate={stats['cap_rate']:.3f}")
    print(f"edges out of graph = {bad}")
    print(f"breadth median  synth={np.median(syn_breadth):.0f}  real={np.median(real_breadth):.0f}")
    print(f"breadth max     synth={max(syn_breadth)}  real={max(real_breadth)}")

    print("\n=== DAY 2 EXIT CRITERIA (SPEC §3 DAY 2, v1.2) ===")
    print(f"[{'x' if secs < 60 else ' '}] 10k campaigns generate in under a minute "
          f"({secs:.1f}s for {len(campaigns):,})")
    print(f"[{'x' if bad == 0 else ' '}] every generated edge exists in the graph "
          f"(violations={bad})")
    print(f"[{'x' if stats['cap_rate'] < 0.2 else ' '}] foothold-cap rate under control "
          f"({stats['cap_rate']:.3f})")
    print("[ ] breadth + credential-reuse distributions visually overlap the fit split "
          "(inspect the medians/max above; full overlay plots on Day 4)")


if __name__ == "__main__":
    main()
