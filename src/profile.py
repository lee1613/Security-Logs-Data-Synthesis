import json

import numpy as np
import pandas as pd


def sessionize(df, gap_seconds):
    """Global time-gap sessionization: a gap > gap_seconds starts a new campaign."""
    df = df.sort_values("time").reset_index(drop=True)
    df["campaign_id"] = (df["time"].diff() > gap_seconds).cumsum().astype(int)
    return df


def freeze_split(df, fit_fraction):
    """Whole-campaign split: order campaigns by start, cut at the inter-campaign
    gap nearest the fit_fraction-of-timeline mark. No campaign is sliced."""
    camp = (df.groupby("campaign_id")["time"]
              .agg(cmin="min", cmax="max").sort_values("cmin").reset_index())
    if len(camp) < 2:
        return df.copy(), df.iloc[0:0].copy()

    t0, t1 = camp["cmin"].iloc[0], camp["cmax"].iloc[-1]
    cut = t0 + fit_fraction * (t1 - t0)

    best_k, best_d = 0, None
    for k in range(len(camp) - 1):
        lo, hi = camp["cmax"].iloc[k], camp["cmin"].iloc[k + 1]
        d = 0 if lo <= cut <= hi else min(abs(cut - lo), abs(cut - hi))
        if best_d is None or d < best_d:
            best_d, best_k = d, k

    fit_ids = set(camp["campaign_id"].iloc[: best_k + 1])
    fit = df[df["campaign_id"].isin(fit_ids)].copy()
    holdout = df[~df["campaign_id"].isin(fit_ids)].copy()
    return fit, holdout


def _hop_bucket(i):
    if i == 0:
        return "0"
    if i <= 2:
        return "1-2"
    return "3+"


def build_profile(fit_df, graph):
    """All Day-1 distributions, fitted on the FIT split only. JSON-serializable."""
    fit = fit_df.sort_values(["campaign_id", "time"]).reset_index(drop=True)
    fit["hop_index"] = fit.groupby("campaign_id").cumcount()

    chain_lengths = fit.groupby("campaign_id").size().tolist()

    dts = []
    for _, grp in fit.groupby("campaign_id"):
        d = grp["time"].diff().dropna()
        dts.extend(int(x) for x in d[d > 0].tolist())

    seeds = fit[fit["hop_index"] == 0]
    seed_hosts = {
        "out_degree": [int(graph.nodes[h]["out_degree"]) for h in seeds["src_computer"] if h in graph],
        "in_degree": [int(graph.nodes[h]["in_degree"]) for h in seeds["src_computer"] if h in graph],
    }

    target_indegree = [int(graph.nodes[t]["in_degree"]) for t in fit["dst_computer"] if t in graph]

    cond = {}
    for _, row in fit.iterrows():
        dst = row["dst_computer"]
        is_srv = bool(graph.nodes[dst]["is_server"]) if dst in graph else False
        key = f"{is_srv}|{_hop_bucket(int(row['hop_index']))}|{row['auth_orientation']}"
        val = f"{row['auth_type']}|{row['logon_type']}|{row['success']}"
        cond.setdefault(key, {}).setdefault(val, 0)
        cond[key][val] += 1

    reuse = []
    for _, grp in fit.groupby("campaign_id"):
        seen, firsts = set(), []
        for i, u in enumerate(grp["src_user"].tolist()):
            if u not in seen:
                seen.add(u)
                firsts.append(i)
        reuse.append({"n_distinct_credentials": len(seen),
                      "first_appearance_hops": firsts})

    return {
        "chain_length": {
            "values": chain_lengths,
            "median": float(np.median(chain_lengths)) if chain_lengths else 0.0,
        },
        "inter_hop_dt": {"values": dts},
        "seed_hosts": seed_hosts,
        "target_indegree": target_indegree,
        "conditional_attributes": cond,
        "credential_reuse": reuse,
    }


def save_profile(profile, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
