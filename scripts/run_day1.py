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

    # Two arms, same code path. `full` describes the corpus; `fit` is what every
    # model is allowed to see (see parse._window). Building both here is what
    # makes the leak contrast in the validation report reproducible.
    t_fit = cfg["day3"]["fit_window"][1]
    arms = {"full": (None, p["aggregates"], p["graph"]),
            "fit": (t_fit, p["aggregates_fit"], p["graph_fit"])}
    graphs = {}
    for name, (t_hi, agg_path, graph_path) in arms.items():
        print(f"[1/6] aggregates ({name}, t_hi={t_hi}) ...")
        agg = {"edges": P.compute_edge_counts(con, p["auth_gz"], t_hi),
               "user_host": P.compute_user_host_counts(con, p["auth_gz"], t_hi),
               "hourly": P.compute_hourly_volume(con, p["auth_gz"], t_hi),
               "marginals": P.compute_marginals(con, p["auth_gz"], t_hi)}
        with open(agg_path, "wb") as f:
            pickle.dump(agg, f)

        print(f"[2/6] graph ({name}) ...")
        graphs[name] = P.build_graph(agg["edges"],
                                     cfg["graph"]["server_in_degree_percentile"],
                                     cfg["graph"]["server_min_in_out_ratio"])
        P.save_graph(graphs[name], graph_path)
        events = int(agg["edges"]["weight"].sum())
        print(f"    nodes={graphs[name].number_of_nodes()} "
              f"edges={graphs[name].number_of_edges()} events={events:,}"
              + ("  (SPEC §2.2 expects ~17,684 computers)" if name == "full" else ""))
    g = graphs["full"]

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
    prof = PR.build_profile(fit, graphs["fit"])   # generator input => fit arm
    PR.save_profile(prof, p["profile"])
    print(f"    chain_length median={prof['chain_length']['median']}")

    print("\n=== DAY 1 EXIT CRITERIA ===")
    both = all(os.path.exists(p[k]) for k in ("graph", "graph_fit"))
    print(f"[{'x' if both else ' '}] graph.pkl + graph_fit.pkl written; "
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
