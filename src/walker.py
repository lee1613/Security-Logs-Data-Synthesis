"""Day-2 credential-reuse FAN-OUT generator (SPEC v1.2).

The labeled LANL red-team activity is fan-out from a foothold to many real
targets, NOT a multi-hop path (chained_links = 0 in all 19 campaigns). This
module seeds a foothold, harvests its credentials, samples a fan-out breadth,
and draws that many real out-edge targets weighted by edge weight, target
in-degree, and a credential bonus. Attributes are constant and attached by the
Day-3 writer, not here.
"""
import pickle

import numpy as np
import pandas as pd
import networkx as nx


def fit_fanout_distributions(fit_df):
    """Reduce the fit split to the sampling distributions the generator needs.
    breadth = distinct targets per campaign (the fan-out driver);
    creds_per_campaign = distinct source users per campaign (Day-4 fidelity);
    inter_event_dt = positive intra-campaign time gaps (Δt bootstrap)."""
    g = fit_df.groupby("campaign_id")
    breadth = g["dst_computer"].nunique().tolist()
    creds = g["src_user"].nunique().tolist()
    dts = []
    for _, grp in g:
        d = grp.sort_values("time")["time"].diff().dropna()
        dts.extend(int(x) for x in d[d > 0].tolist())
    return {
        "breadth": breadth,
        "creds_per_campaign": creds,
        "inter_event_dt": dts or [1],   # never empty; Δt must be > 0
    }


def build_host_users(user_host_df):
    """Invert the user_host aggregate into host -> set(users seen on it)."""
    m = {}
    for u, c in zip(user_host_df["user"], user_host_df["computer"]):
        m.setdefault(c, set()).add(u)
    return m


def foothold_candidates(graph, min_out_degree):
    """Precompute (once) the foothold pool and out-degree-proportional
    probabilities. Called by the corpus loop, not per campaign."""
    cands = [n for n in graph.nodes if graph.out_degree(n) >= min_out_degree]
    w = np.array([graph.out_degree(n) for n in cands], dtype=float)
    cp = w / w.sum()
    return cands, cp


def pick_foothold(cands, cp, rng):
    return cands[rng.choice(len(cands), p=cp)]


def harvest_credentials(foothold, host_users, max_creds, rng):
    """Credentials observed on the foothold, capped. Empty set if none."""
    creds = list(host_users.get(foothold, set()))
    if len(creds) > max_creds:
        idx = rng.choice(len(creds), size=max_creds, replace=False)
        creds = [creds[i] for i in idx]
    return set(creds)
