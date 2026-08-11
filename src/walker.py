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


def target_weights(graph, foothold, targets, host_users, compromised,
                   alpha, beta, credential_bonus):
    """Target selection probabilities over a foothold's REAL out-edges:
    edge_weight^alpha * (in_degree+1)^beta * credential_bonus, normalized.
    With alpha=beta=0 and credential_bonus=1.0 every weight is 1.0 and the vector
    is exactly uniform -- that is Day-4 baseline #2's naive fan-out."""
    weights = np.empty(len(targets), dtype=float)
    for i, t in enumerate(targets):
        edge_w = float(graph[foothold][t]["weight"])
        indeg = float(graph.nodes[t]["in_degree"])
        bonus = credential_bonus if (host_users.get(t, set()) & compromised) else 1.0
        weights[i] = (edge_w ** alpha) * ((indeg + 1.0) ** beta) * bonus
    return weights / weights.sum()


def generate_campaign(graph, host_users, dists, cands, cp, rng,
                      alpha, beta, credential_bonus, max_creds, naive=False):
    """One fan-out campaign. Returns (events, capped) where events is a list of
    (dt_offset, user, foothold, target); or (None, False) to signal a discard.

    `naive` is the Day-4 baseline #2 switch and is OFF by default -- with it off
    this function is byte-identical to the pre-Day-4 generator. On, it drops the
    two CREDENTIAL-layer behaviours (the creds-per-campaign restriction and the
    preference for a user already seen on the target) while leaving footholds,
    breadth and the V1 real-edge constraint alone. The WEIGHTING layer is turned
    off by the caller passing alpha=beta=0 and credential_bonus=1.0."""
    foothold = pick_foothold(cands, cp, rng)
    compromised = harvest_credentials(foothold, host_users, max_creds, rng)
    targets = list(graph.successors(foothold))
    if not compromised or not targets:
        return None, False

    if not naive:
        # gap #2 fix: reuse-intensity matches the fit split — restrict to m sampled creds
        m = int(rng.choice(dists["creds_per_campaign"]))
        comp_list = list(compromised)
        if 0 < m < len(comp_list):
            idx = rng.choice(len(comp_list), size=m, replace=False)
            compromised = {comp_list[i] for i in idx}

    k = int(rng.choice(dists["breadth"]))
    capped = k > len(targets)
    k = min(k, len(targets))

    p = target_weights(graph, foothold, targets, host_users, compromised,
                       alpha, beta, credential_bonus)
    idx = rng.choice(len(targets), size=k, replace=False, p=p)
    chosen = [targets[i] for i in idx]

    dts = rng.choice(dists["inter_event_dt"], size=k)
    offsets = np.cumsum(dts)

    events = []
    for off, t in zip(offsets, chosen):
        pool = (list(compromised) if naive
                else list(host_users.get(t, set()) & compromised) or list(compromised))
        user = pool[rng.integers(len(pool))]
        events.append((int(off), user, foothold, t))
    return events, capped


def generate_corpus(graph, host_users, dists, n, alpha, beta,
                    credential_bonus, min_out_degree, max_creds, seed,
                    naive=False):
    """Generate n valid fan-out campaigns. Foothold pool precomputed once.
    Returns (campaigns, stats) with discard_rate and cap_rate logged.
    `naive` is passed straight through to generate_campaign (baseline #2)."""
    rng = np.random.default_rng(seed)
    cands, cp = foothold_candidates(graph, min_out_degree)
    if not cands:
        raise ValueError(f"no foothold has out-degree >= {min_out_degree}")

    campaigns, discards, caps, attempts = [], 0, 0, 0
    max_attempts = n * 10  # ponytail: bounded so a pathological graph can't spin forever
    while len(campaigns) < n and attempts < max_attempts:
        attempts += 1
        events, capped = generate_campaign(
            graph, host_users, dists, cands, cp, rng,
            alpha, beta, credential_bonus, max_creds, naive=naive)
        if events is None:
            discards += 1
            continue
        if capped:
            caps += 1
        campaigns.append(events)

    stats = {
        "n": len(campaigns),
        "attempts": attempts,
        "discard_rate": discards / max(attempts, 1),
        "cap_rate": caps / max(len(campaigns), 1),
    }
    return campaigns, stats


def save_campaigns(campaigns, path):
    with open(path, "wb") as f:
        pickle.dump(campaigns, f)
