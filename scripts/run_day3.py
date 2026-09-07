import pickle
import time

import numpy as np
import pandas as pd

from src.config import load_config
import src.parse as P
import src.walker as W
import src.writer as Wr
import src.benign as B
import src.mixed as M


def main():
    cfg = load_config()
    p, fo, d3 = cfg["paths"], cfg["fanout"], cfg["day3"]

    graph = P.load_graph(p["graph_fit"])   # generator input => fit arm
    with open(p["aggregates"], "rb") as f:
        agg = pickle.load(f)
    fit = pd.read_csv(p["redteam_fit"])
    full = pd.read_csv(p["redteam_full"])

    dists = W.fit_fanout_distributions(fit)
    host_users = W.build_host_users(agg["user_host"])
    hourly = agg["hourly"]

    # 1) regenerate campaigns (now with the creds-per-campaign fix)
    campaigns, stats = W.generate_corpus(
        graph, host_users, dists, n=fo["n_campaigns"],
        alpha=cfg["alpha"], beta=cfg["beta"],
        credential_bonus=fo["credential_bonus"],
        min_out_degree=fo["min_foothold_out_degree"],
        max_creds=fo["max_harvest_creds"], seed=cfg["seed"])

    # 2) place on the timeline + emit rows/labels
    rng = np.random.default_rng(cfg["seed"])
    coll = d3["collection_seconds"]
    all_rows, all_labels, rid = [], [], 0
    for cid, camp in enumerate(campaigns):
        max_off = camp[-1][0]
        start = Wr.sample_start_time(hourly, rng, max_offset=max_off, collection_seconds=coll)
        placed = Wr.place_campaign(camp, start)
        rows, labels = Wr.campaign_to_rows(placed, campaign_id=cid, row_id_start=rid)
        all_rows.extend(rows); all_labels.extend(labels); rid += len(rows)
    Wr.write_auth_csv(all_rows, d3["synth_auth"])
    Wr.write_labels_csv(all_labels, d3["synth_labels"])
    mal_df = pd.DataFrame(all_rows)

    # 3) windowed benign (fit -> train negatives, holdout -> test negatives)
    redteam_edges = set(zip(full["src_computer"], full["dst_computer"]))
    t0 = time.time()
    ben_fit = B.sample_benign(p["auth_gz"], *d3["fit_window"], n=d3["benign_n"],
                              redteam_edges=redteam_edges, seed=cfg["seed"])
    ben_hold = B.sample_benign(p["auth_gz"], *d3["holdout_window"], n=d3["benign_n"],
                               redteam_edges=redteam_edges, seed=cfg["seed"] + 1)
    ben_fit.to_csv(d3["benign_fit"], index=False)
    ben_hold.to_csv(d3["benign_holdout"], index=False)
    print(f"benign extraction: {time.time()-t0:.1f}s  fit={len(ben_fit):,}  holdout={len(ben_hold):,}")

    # 4) mixed corpora at the sweep rates (malicious into fit-window benign)
    achieved = {}
    for br in d3["base_rates"]:
        mixed, labels = M.build_mixed_corpus(mal_df, ben_fit, rate=br["rate"], seed=cfg["seed"])
        Wr.write_auth_csv(mixed.to_dict("records"),
                          d3["synth_auth"].replace("synth_auth.csv", f"mixed_{br['name']}.csv"))
        labels.to_csv(d3["synth_labels"].replace("synth_labels.csv",
                                                 f"mixed_{br['name']}_labels.csv"), index=False)
        n_m = int((labels.is_malicious == 1).sum()); n_b = int((labels.is_malicious == 0).sum())
        achieved[br["name"]] = n_m / n_b
        print(f"mixed {br['name']}: {len(mixed):,} rows  mal={n_m:,}  ben={n_b:,}  "
              f"rate={n_m / n_b:.5f} (target {br['rate']})")

    # ---- exit criteria (SPEC §3 DAY 3) ----
    reread = pd.read_csv(d3["synth_auth"], header=None, names=P.AUTH_COLS)
    parses = list(reread.columns) == list(P.AUTH_COLS) and len(reread) == len(all_rows)
    aligned = len(pd.read_csv(d3["synth_labels"])) == len(all_rows)
    # field-domain subset check on the constant categoricals
    dom_ok = (set(reread["auth_type"]) <= {"NTLM"} and set(reread["logon_type"]) <= {"Network"}
              and set(reread["success"]) <= {"Success"})
    # every emitted edge is real (V1 preview)
    bad_edges = sum(1 for r in all_rows if not graph.has_edge(r["src_computer"], r["dst_computer"]))
    # temporal footprint resembles real red team (business hours)
    hod = (reread["time"] // 3600) % 24
    biz_frac = float(((hod >= 6) & (hod <= 17)).mean())

    print("\n=== DAY 3 EXIT CRITERIA (SPEC §3 DAY 3) ===")
    print(f"[{'x' if parses else ' '}] output parses with the real auth reader ({len(reread):,} rows)")
    print(f"[{'x' if dom_ok else ' '}] categorical value domains subset of real (NTLM/Network/Success)")
    print(f"[{'x' if aligned else ' '}] labels row-aligned to data ({len(all_rows):,} rows)")
    print(f"[{'x' if bad_edges == 0 else ' '}] every emitted (src,dst) is a real edge (violations={bad_edges})")
    # mixed sweep ok = each corpus within 20% of its target rate (distinct base rates achieved)
    sweep_ok = all(abs(achieved[b["name"]] - b["rate"]) <= 0.2 * b["rate"] for b in d3["base_rates"])
    print(f"[{'x' if sweep_ok else ' '}] mixed corpora at distinct target rates "
          f"{ {b['name']: round(achieved[b['name']], 5) for b in d3['base_rates']} }")
    print(f"[i] business-hours (6-17) fraction of synthetic events = {biz_frac:.2f} (real red team ~1.0)")
    print(f"[i] discard_rate={stats['discard_rate']:.3f} cap_rate={stats['cap_rate']:.3f}")


if __name__ == "__main__":
    main()
