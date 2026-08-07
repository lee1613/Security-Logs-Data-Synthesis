import json
import os
import pickle
import stat

import duckdb

from src.config import load_config
import src.parse as P
import src.profile as PR


def _freeze_readonly(path):
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def main():
    cfg = load_config()
    p = cfg["paths"]
    con = P.open_con()

    print("[1/6] aggregates ...")
    edges = P.compute_edge_counts(con, p["auth_gz"])
    user_host = P.compute_user_host_counts(con, p["auth_gz"])
    hourly = P.compute_hourly_volume(con, p["auth_gz"])
    marginals = P.compute_marginals(con, p["auth_gz"])
    with open(p["aggregates"], "wb") as f:
        pickle.dump({"edges": edges, "user_host": user_host,
                     "hourly": hourly, "marginals": marginals}, f)

    print("[2/6] graph ...")
    g = P.build_graph(edges,
                      cfg["graph"]["server_in_degree_percentile"],
                      cfg["graph"]["server_min_in_out_ratio"])
    P.save_graph(g, p["graph"])
    print(f"    nodes={g.number_of_nodes()} edges={g.number_of_edges()} "
          f"(SPEC §2.2 expects ~17,684 computers)")

    print("[3/6] recover redteam ...")
    full, unmatched = P.recover_redteam(con, p["auth_gz"], p["redteam_gz"])
    full.to_csv(p["redteam_full"], index=False)
    print(f"    matched={len(full)} unmatched={unmatched}")

    print("[4/6] target-attractiveness validation ...")
    val = P.validate_target_attractiveness(g, full, cfg["graph"]["target_keep_ratio"])
    with open(p["target_validation"], "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2)
    print(f"    keep_server_proxy={val['keep_server_proxy']} "
          f"(target/all in-degree median ratio={val['separation_ratio']:.2f})")

    print("[5/6] sessionize + freeze split ...")
    sess = PR.sessionize(full, cfg["campaign_gap_seconds"])
    fit, holdout = PR.freeze_split(sess, cfg["split"]["fit_fraction"])
    fit.to_csv(p["redteam_fit"], index=False)
    holdout.to_csv(p["redteam_holdout"], index=False)
    _freeze_readonly(p["redteam_fit"])
    _freeze_readonly(p["redteam_holdout"])
    print(f"    campaigns={sess['campaign_id'].nunique()} "
          f"fit={len(fit)} holdout={len(holdout)}")

    print("[6/6] profile fit split ...")
    prof = PR.build_profile(fit, g)
    PR.save_profile(prof, p["profile"])
    print(f"    chain_length median={prof['chain_length']['median']}")

    print("\n=== DAY 1 EXIT CRITERIA ===")
    print(f"[{'x' if os.path.exists(p['graph']) else ' '}] graph.pkl written; "
          f"nodes/edges printed above")
    print(f"[{'x' if unmatched == 0 else ' '}] all redteam joined "
          f"(unmatched={unmatched} — count & explain any)")
    ro = not os.access(p["redteam_fit"], os.W_OK)
    print(f"[{'x' if ro else ' '}] split files written and immutable")
    need = {"chain_length", "inter_hop_dt", "seed_hosts", "target_indegree",
            "conditional_attributes", "credential_reuse"}
    print(f"[{'x' if need <= set(prof) else ' '}] profile contains every distribution")


if __name__ == "__main__":
    main()
