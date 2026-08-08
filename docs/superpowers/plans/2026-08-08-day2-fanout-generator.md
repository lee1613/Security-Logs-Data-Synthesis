# Day 2 — Credential-Reuse Fan-Out Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/walker.py` — a credential-aware **fan-out** generator that, from a sampled foothold host, emits many authentications outward to real graph targets, reproducing the LANL red team's measured structure (one source → many high-in-degree targets, credential reuse), with a hard "every edge is real" constraint.

**Architecture:** The labeled red-team activity is **fan-out from a foothold, not a multi-hop path** (`chained_links = 0` in all 19 campaigns — SPEC v1.2, `docs/superpowers/phase-gate-day1-to-day2.md`). So the generator does NOT walk a path. It: seeds a foothold (sampled ∝ out-degree), harvests the credentials seen on it, samples a **fan-out breadth `k`** from the empirical per-campaign distinct-target distribution, then draws `k` real out-edge targets weighted by `edge_weight^alpha × (in_degree+1)^beta × credential_bonus`. Attributes are constant `(NTLM, Network, LogOn, Success)` and are attached by the Day-3 writer, not here. All sampling distributions are derived at load time from the frozen `redteam_fit.csv` (the source of truth) plus `graph.pkl` and the `user_host` aggregate.

**Tech Stack:** Python 3.11, NetworkX, NumPy, Pandas, PyYAML, pytest. Reuses Day-1 artifacts: `data/derived/graph.pkl`, `data/derived/aggregates.pkl` (`user_host`), `data/derived/redteam_fit.csv`.

**Scope:** Day 2 only. Exit criteria = SPEC §3 DAY 2 (v1.2). Do **not** roll into Day 3 — the SPEC §3.0 phase-gate grilling checkpoint sits between them. Time placement + LANL-format writing + benign interleave are Day 3.

**SPEC v1.2 decisions this plan encodes (do not relitigate):**
- Structure = **fan-out**, not path. Sample **breadth** (distinct targets/campaign), never "chain length / hops".
- Hard constraint: every `(foothold, target)` is a real graph out-edge. Credential realism is a **weight bonus, not a filter**.
- Foothold ∝ out-degree; targets ∝ `edge_weight^alpha × (in_degree+1)^beta × credential_bonus`.
- Attributes are constant — attached downstream (Day 3), not modeled here.
- Distributions fitted on the **fit split only**. Holdout stays untouched until Day 4.

---

## File Structure

| File | Responsibility |
|---|---|
| `config.yaml` | Add a `fanout:` block (foothold threshold, harvest cap, credential bonus, campaign count). `alpha`/`beta` already present. |
| `src/walker.py` | Fan-out generator: derive distributions, build host→users index, pick foothold, harvest creds, generate one campaign, generate a corpus with discard/cap logging. |
| `scripts/run_day2.py` | Load Day-1 artifacts, generate the corpus at scale, assert V1 edge-validity, print exit-criteria checklist (timing, breadth overlap, discard/cap rates), save `data/derived/synth_campaigns.pkl`. |
| `tests/test_walker.py` | Unit tests for every `walker.py` function against a tiny hand-built graph + fit frame. |

**Data produced at runtime:** `data/derived/synth_campaigns.pkl` — a list of campaigns, each a list of `(dt_offset, user, src_host, dst_host)` tuples. (Day 3 consumes this.)

**Skipped deliberately** (`ponytail:`): optional shallow second-tier depth (a target promoted to a sub-foothold) — the data shows zero multi-hop, so default off; add only if real multi-tier structure is later found (SPEC §7 Future Work). No per-edge `(user,src,dst)` credential index — too sparse (SPEC §3 DAY 2); the bonus uses host-level `user_host` membership.

---

### Task 1: Add the `fanout` config block

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Append the `fanout` block to `config.yaml`**

Add these lines at the end of `config.yaml` (leave existing `alpha: 1.0` / `beta: 1.0` as-is — the generator reads them):

```yaml
# Day-2 fan-out generator knobs
fanout:
  min_foothold_out_degree: 25   # a foothold must have >= this many real out-edges
                                # (smallest real red-team seed out-degree ~29)
  max_harvest_creds: 60         # cap credentials pulled from a foothold (real max ~45)
  credential_bonus: 2.0         # multiplier for a target where a compromised cred is seen
  n_campaigns: 10000            # corpus size for the Day-2 scale run / exit criterion
```

- [ ] **Step 2: Verify it parses**

Run:
```bash
python -c "from src.config import load_config; c=load_config(); print(c['fanout'], c['alpha'], c['beta'])"
```
Expected: the `fanout` dict prints plus `1.0 1.0`.

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "chore: day-2 fan-out config knobs"
```

---

### Task 2: Derive fan-out distributions + host→users index

**Files:**
- Create: `src/walker.py`
- Test: `tests/test_walker.py`

`fit_fanout_distributions` reduces the fit frame to the three sampling distributions the generator needs: **breadth** (distinct targets per campaign), **creds_per_campaign** (distinct source users per campaign — a descriptive fidelity check for Day 4), and **inter_event_dt** (positive intra-campaign time gaps to bootstrap Δt). `build_host_users` inverts the `user_host` aggregate into `host -> set(users)` for harvest + credential-bonus lookups.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
import networkx as nx
import pytest
import src.walker as W


def _fit(rows):
    return pd.DataFrame(rows, columns=[
        "time", "src_user", "dst_user", "src_computer", "dst_computer",
        "auth_type", "logon_type", "auth_orientation", "success", "campaign_id"])


def test_fit_fanout_distributions():
    # campaign 0: foothold C1 -> C2,C3,C4 (breadth 3), users U1,U2 (2 creds), dts 60,60
    # campaign 1: foothold C1 -> C3 (breadth 1), user U1 (1 cred), no dt
    fit = _fit([
        [0,   "U1@D", "U1@D", "C1", "C2", "NTLM", "Network", "LogOn", "Success", 0],
        [60,  "U2@D", "U2@D", "C1", "C3", "NTLM", "Network", "LogOn", "Success", 0],
        [120, "U1@D", "U1@D", "C1", "C4", "NTLM", "Network", "LogOn", "Success", 0],
        [100000, "U1@D", "U1@D", "C1", "C3", "NTLM", "Network", "LogOn", "Success", 1],
    ])
    d = W.fit_fanout_distributions(fit)
    assert sorted(d["breadth"]) == [1, 3]
    assert sorted(d["creds_per_campaign"]) == [1, 2]
    assert sorted(d["inter_event_dt"]) == [60, 60]
    assert all(x > 0 for x in d["inter_event_dt"])


def test_build_host_users():
    uh = pd.DataFrame(
        [["U1@D", "C1", 5], ["U2@D", "C1", 2], ["U1@D", "C2", 1]],
        columns=["user", "computer", "cnt"])
    m = W.build_host_users(uh)
    assert m["C1"] == {"U1@D", "U2@D"}
    assert m["C2"] == {"U1@D"}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_walker.py -q`
Expected: FAIL — `AttributeError: module 'src.walker' has no attribute 'fit_fanout_distributions'`.

- [ ] **Step 3: Write `src/walker.py` (distributions + index)**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_walker.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/walker.py tests/test_walker.py
git commit -m "feat: derive fan-out distributions and host->users index"
```

---

### Task 3: Foothold selection + credential harvest

**Files:**
- Modify: `src/walker.py`
- Test: `tests/test_walker.py`

`pick_foothold` samples a foothold from precomputed candidates (nodes with out-degree ≥ threshold) with probability ∝ out-degree — big operator hosts get picked more, matching the real C17693-dominant pattern. Candidates/probabilities are precomputed **once** by the corpus loop (Task 5), not per campaign, so 10k campaigns stay fast. `harvest_credentials` pulls the users seen on the foothold, capped.

- [ ] **Step 1: Add the failing tests**

```python
def _rng():
    return np.random.default_rng(42)


def test_pick_foothold_prefers_high_out_degree():
    g = nx.DiGraph()
    g.add_edge("F", "A"); g.add_edge("F", "B"); g.add_edge("F", "C")  # out 3
    g.add_edge("S", "A")                                              # out 1
    cands, cp = W.foothold_candidates(g, min_out_degree=1)
    assert set(cands) == {"F", "S"}
    # F has 3x the out-degree of S -> 0.75 vs 0.25 probability
    pF = cp[cands.index("F")]
    assert abs(pF - 0.75) < 1e-9
    pick = W.pick_foothold(cands, cp, _rng())
    assert pick in cands


def test_foothold_candidates_threshold_excludes_small():
    g = nx.DiGraph()
    g.add_edge("F", "A"); g.add_edge("F", "B")  # out 2
    g.add_edge("S", "A")                         # out 1
    cands, _ = W.foothold_candidates(g, min_out_degree=2)
    assert cands == ["F"]


def test_harvest_credentials_caps():
    host_users = {"F": {f"U{i}@D" for i in range(10)}}
    got = W.harvest_credentials("F", host_users, max_creds=4, rng=_rng())
    assert len(got) == 4
    assert got <= host_users["F"]
    # no creds seen -> empty set
    assert W.harvest_credentials("X", host_users, max_creds=4, rng=_rng()) == set()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_walker.py::test_pick_foothold_prefers_high_out_degree -q`
Expected: FAIL — `has no attribute 'foothold_candidates'`.

- [ ] **Step 3: Add foothold + harvest functions to `src/walker.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_walker.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/walker.py tests/test_walker.py
git commit -m "feat: foothold selection and credential harvest"
```

---

### Task 4: Generate one fan-out campaign

**Files:**
- Modify: `src/walker.py`
- Test: `tests/test_walker.py`

`generate_campaign` produces one campaign: pick foothold, harvest creds, sample breadth `k`, draw `k` real out-edge targets (weighted, without replacement, capped at real out-degree), assign each a compromised credential seen on that target (backoff: any compromised cred), and cumulative strictly-increasing Δt offsets. Returns `(events, capped)`, or `(None, False)` to signal a discard (no creds or no targets).

- [ ] **Step 1: Add the failing tests**

```python
def _graph_with_targets():
    g = nx.DiGraph()
    for dst, w in [("T1", 5), ("T2", 1), ("T3", 3)]:
        g.add_edge("F", dst, weight=w)
    # set in_degree node attrs the weighting reads
    for n in g.nodes:
        g.nodes[n]["in_degree"] = {"T1": 10, "T2": 2, "T3": 7}.get(n, 0)
    return g


def test_generate_campaign_valid_edges_and_creds():
    g = _graph_with_targets()
    host_users = {"F": {"U1@D", "U2@D"}, "T1": {"U1@D"}, "T2": {"U9@D"}, "T3": {"U2@D"}}
    dists = {"breadth": [3], "creds_per_campaign": [2], "inter_event_dt": [60, 90]}
    cands, cp = W.foothold_candidates(g, min_out_degree=1)  # only F qualifies
    events, capped = W.generate_campaign(
        g, host_users, dists, cands, cp, _rng(),
        alpha=1.0, beta=1.0, credential_bonus=2.0, max_creds=60)
    assert capped is False
    assert len(events) == 3                       # breadth 3, F has 3 targets
    for off, user, src, dst in events:
        assert src == "F"
        assert g.has_edge(src, dst)               # V1: every edge real
        assert user in {"U1@D", "U2@D"}           # only compromised creds used
        assert off > 0
    # strictly increasing offsets (Δt > 0)
    offs = [e[0] for e in events]
    assert offs == sorted(offs) and len(set(offs)) == len(offs)


def test_generate_campaign_caps_breadth_at_out_degree():
    g = _graph_with_targets()  # F has 3 targets
    host_users = {"F": {"U1@D"}, "T1": {"U1@D"}, "T2": {"U1@D"}, "T3": {"U1@D"}}
    dists = {"breadth": [10], "creds_per_campaign": [1], "inter_event_dt": [60]}
    cands, cp = W.foothold_candidates(g, min_out_degree=1)
    events, capped = W.generate_campaign(
        g, host_users, dists, cands, cp, _rng(),
        alpha=1.0, beta=1.0, credential_bonus=2.0, max_creds=60)
    assert capped is True
    assert len(events) == 3                        # capped at out-degree 3


def test_generate_campaign_discards_when_no_creds():
    g = _graph_with_targets()
    host_users = {"T1": {"U1@D"}}                  # F has no creds seen
    dists = {"breadth": [2], "creds_per_campaign": [1], "inter_event_dt": [60]}
    cands, cp = W.foothold_candidates(g, min_out_degree=1)
    events, capped = W.generate_campaign(
        g, host_users, dists, cands, cp, _rng(),
        alpha=1.0, beta=1.0, credential_bonus=2.0, max_creds=60)
    assert events is None and capped is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_walker.py::test_generate_campaign_valid_edges_and_creds -q`
Expected: FAIL — `has no attribute 'generate_campaign'`.

- [ ] **Step 3: Add `generate_campaign` to `src/walker.py`**

```python
def generate_campaign(graph, host_users, dists, cands, cp, rng,
                      alpha, beta, credential_bonus, max_creds):
    """One fan-out campaign. Returns (events, capped) where events is a list of
    (dt_offset, user, foothold, target); or (None, False) to signal a discard."""
    foothold = pick_foothold(cands, cp, rng)
    compromised = harvest_credentials(foothold, host_users, max_creds, rng)
    targets = list(graph.successors(foothold))
    if not compromised or not targets:
        return None, False

    k = int(rng.choice(dists["breadth"]))
    capped = k > len(targets)
    k = min(k, len(targets))

    # candidate weights: edge_weight^alpha * (in_degree+1)^beta * credential_bonus
    weights = np.empty(len(targets), dtype=float)
    for i, t in enumerate(targets):
        edge_w = float(graph[foothold][t]["weight"])
        indeg = float(graph.nodes[t]["in_degree"])
        bonus = credential_bonus if (host_users.get(t, set()) & compromised) else 1.0
        weights[i] = (edge_w ** alpha) * ((indeg + 1.0) ** beta) * bonus
    p = weights / weights.sum()

    idx = rng.choice(len(targets), size=k, replace=False, p=p)
    chosen = [targets[i] for i in idx]

    dts = rng.choice(dists["inter_event_dt"], size=k)
    offsets = np.cumsum(dts)

    events = []
    for off, t in zip(offsets, chosen):
        pool = list(host_users.get(t, set()) & compromised) or list(compromised)
        user = pool[rng.integers(len(pool))]
        events.append((int(off), user, foothold, t))
    return events, capped
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_walker.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/walker.py tests/test_walker.py
git commit -m "feat: generate one credential-reuse fan-out campaign"
```

---

### Task 5: Generate a corpus with discard/cap logging

**Files:**
- Modify: `src/walker.py`
- Test: `tests/test_walker.py`

`generate_corpus` precomputes the foothold pool once, then loops until `n` valid campaigns exist, counting discards (empty creds/targets) and caps (breadth clipped to out-degree). It returns the campaigns plus a `stats` dict so the SPEC guard rails (discard rate, cap rate) are visible.

- [ ] **Step 1: Add the failing test**

```python
def test_generate_corpus_stats_and_validity():
    g = _graph_with_targets()
    host_users = {"F": {"U1@D", "U2@D"}, "T1": {"U1@D"}, "T2": {"U9@D"}, "T3": {"U2@D"}}
    dists = {"breadth": [2, 3], "creds_per_campaign": [2], "inter_event_dt": [60, 90]}
    campaigns, stats = W.generate_corpus(
        g, host_users, dists, n=50, alpha=1.0, beta=1.0,
        credential_bonus=2.0, min_out_degree=1, max_creds=60, seed=42)
    assert len(campaigns) == 50
    assert stats["n"] == 50
    assert 0.0 <= stats["discard_rate"] <= 1.0
    assert 0.0 <= stats["cap_rate"] <= 1.0
    # V1: every generated edge exists in the graph
    for camp in campaigns:
        for _, _, src, dst in camp:
            assert g.has_edge(src, dst)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_walker.py::test_generate_corpus_stats_and_validity -q`
Expected: FAIL — `has no attribute 'generate_corpus'`.

- [ ] **Step 3: Add `generate_corpus` to `src/walker.py`**

```python
def generate_corpus(graph, host_users, dists, n, alpha, beta,
                    credential_bonus, min_out_degree, max_creds, seed):
    """Generate n valid fan-out campaigns. Foothold pool precomputed once.
    Returns (campaigns, stats) with discard_rate and cap_rate logged."""
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
            alpha, beta, credential_bonus, max_creds)
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_walker.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/walker.py tests/test_walker.py
git commit -m "feat: generate fan-out corpus with discard/cap stats"
```

---

### Task 6: Scale run — generate 10k campaigns, check exit criteria

**Files:**
- Create: `scripts/run_day2.py`

Loads the Day-1 artifacts, derives distributions from the frozen fit split, generates `n_campaigns`, times it, asserts V1 edge-validity on the whole corpus, compares synthetic vs. real fan-out-breadth and creds-per-campaign, and prints the SPEC §3 DAY 2 exit-criteria checklist.

- [ ] **Step 1: Write `scripts/run_day2.py`**

```python
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
```

- [ ] **Step 2: Run the scale generator**

Run: `python -m scripts.run_day2`
Expected: a timing line, `edges out of graph = 0`, `discard_rate` and `cap_rate` printed, and the exit-criteria checklist. Breadth medians should be in the same ballpark as real (real median breadth ≈ 9; note the two real mega-fan-outs 168/88 mean synthetic max breadth is capped by whichever foothold's out-degree is drawn — expected, document if synthetic max is lower).

- [ ] **Step 3: Confirm the whole suite still passes**

Run: `python -m pytest -q`
Expected: PASS (all parse + profile + walker tests).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_day2.py
git commit -m "feat: day-2 scale run with V1 edge-validity and exit checks"
```

---

## Day-2 Exit Criteria (SPEC §3 DAY 2, v1.2) — verify before the phase-gate

- [ ] 10,000 campaigns generate in under a minute.
- [ ] Assertion passes: every generated `(foothold, target)` edge exists in the graph (0 violations).
- [ ] Generated fan-out-breadth distribution visually overlaps the real fit-split breadth (medians/max sane in the run output; full overlay plots are a Day-4 V2 deliverable).
- [ ] Credentials-per-campaign overlaps the fit split (descriptive check).
- [ ] Foothold-cap rate logged and under control (< 0.2).

**Do not start Day 3.** SPEC §3.0 mandates a grilling checkpoint (review Day-2 state, revise Day-3 knobs — time placement, benign interleave, the constant-attribute writer — explicit go/no-go) before the attribute/writer layer.

---

## Self-Review

- **Spec coverage (SPEC §3 DAY 2, v1.2):** Seed foothold → Task 3 (`pick_foothold` ∝ out-degree). Harvest → Task 3 (`harvest_credentials`, capped). Fan-out breadth (replaces chain length) → Task 2 (`fit_fanout_distributions`) + Task 4 (`k` sample). Candidate generation + hard edge constraint → Task 4 (`graph.successors`, asserted in Tasks 4–6). Weighting `edge_weight^alpha × (in_degree+1)^beta × credential_bonus` → Task 4. Credential bonus as weight not filter → Task 4 (`host_users & compromised`). Δt bootstrap → Task 4 (`inter_event_dt`). Guard rails (cap at out-degree, log cap/discard) → Tasks 4–5. Exit criteria (10k < 1 min, edges real, breadth overlap, cap logged) → Task 6. Attributes constant/attached downstream → out of Day-2 scope by design (stated in header).
- **Placeholder scan:** every code step is complete runnable code; the one unchecked exit box (visual overlay) is explicitly a Day-4 V2 deliverable, not a Day-2 TODO.
- **Type consistency:** `fit_fanout_distributions` returns `{breadth, creds_per_campaign, inter_event_dt}`, consumed by `generate_campaign`/`generate_corpus` and `run_day2.py` under the same keys. `foothold_candidates(graph, min_out_degree) -> (cands, cp)` signature identical in Tasks 3–5 and the script. `generate_campaign(graph, host_users, dists, cands, cp, rng, alpha, beta, credential_bonus, max_creds)` — same arg order in Task 4 tests and the Task 5 corpus loop. `generate_corpus(..., min_out_degree, max_creds, seed)` — same in Task 5 test and `run_day2.py`. Campaign tuple shape `(dt_offset, user, src, dst)` consistent across generator, corpus V1 check, and `save_campaigns`. Reuses Day-1 `P.load_graph` and `data/derived/aggregates.pkl` keys (`user_host` cols `user,computer,cnt`) exactly as produced by Day-1.
- **Known ceiling:** `pick_foothold` samples footholds independently of the sampled breadth, so a large breadth drawn onto a modest-out-degree foothold gets capped (logged via `cap_rate`). If `cap_rate` is high, the upgrade path is to sample breadth conditioned on the foothold's out-degree — deferred until the cap rate says it matters (`ponytail:`).
```
