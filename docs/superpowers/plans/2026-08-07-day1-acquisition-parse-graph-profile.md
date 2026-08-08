# Day 1 — Acquisition, Parse, Graph, Profile — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the raw LANL `auth.txt.gz` + `redteam.txt.gz` into the Day-1 artifacts — a `NetworkX` graph, a frozen train/test red-team split, and a `redteam_profile.json` — that everything downstream is fitted on.

**Architecture:** DuckDB streams the ~1B-row gzipped `auth.txt` and runs every aggregate (edge/user-host counts, hourly volume, categorical marginals) plus the red-team↔auth recovery join in C++ SQL. Pandas/NetworkX turn those aggregates into a `DiGraph` with derived `is_server` and target-attractiveness features. Profiling and the whole-campaign timeline split run on the recovered 9-field red-team records. Every function is unit-tested against a tiny hand-crafted gzip fixture, then run once at scale.

**Tech Stack:** Python 3.11, DuckDB, NetworkX, NumPy, Pandas, PyYAML, pytest. Windows (no `zcat`/`awk`).

**Scope:** Day 1 only. Day 1's exit criteria (SPEC §3 DAY 1) are the definition of done for this plan. Do **not** roll into Day 2 — the SPEC §3.0 phase-gate grilling checkpoint sits between them.

**SPEC decisions this plan encodes (do not relitigate — SPEC §3):**
- `campaign_gap = 4h` (14400s), 1h fallback is a config flip.
- Split = **whole-campaign assignment**, cut at the inter-campaign gap nearest the **70%-of-timeline** mark. Never slice a campaign.
- Fit distributions on the **fit split only**. The holdout is untouched until Day 4.
- Target-attractiveness fitted to the **structural** in-degree / in:out profile of red-team targets — never host identities. `is_server` is a *derived* binary, gated by a Day-1 validation (Task 7).
- Credential realism enters as a **weight** on Day 2, not a filter — nothing here filters on it.

---

## File Structure

| File | Responsibility |
|---|---|
| `config.yaml` | All knobs: paths, seed, `campaign_gap_seconds`, split fraction, server thresholds, benchmark size. |
| `.gitignore` | Keep `data/` (raw + derived, ~15 GB) out of git. |
| `requirements.txt` | Pinned deps (adds `duckdb`, otherwise SPEC-verified). |
| `src/config.py` | `load_config()` — one function. |
| `src/parse.py` | DuckDB relations, the five aggregate queries, red-team recovery join, graph build + `is_server`, target-attractiveness validation, save/load. |
| `src/profile.py` | `sessionize`, `freeze_split` (whole-campaign), `build_profile`, `save_profile`. |
| `scripts/benchmark.py` | Time the edge query on the first N rows; extrapolate; fire the §5 time-slice fallback if too slow. |
| `scripts/run_day1.py` | Orchestrator: runs the whole pass on real data, writes artifacts, prints the exit-criteria checklist. |
| `tests/conftest.py` | `mini_auth_path` / `mini_redteam_path` gzip fixtures with hand-computed expected aggregates. |
| `tests/test_parse.py` | Unit tests for every `parse.py` function against the fixture. |
| `tests/test_profile.py` | Unit tests for sessionize / split / profile. |

`data/` layout produced at runtime: `data/raw/` (downloads + `CHECKSUMS.txt`), `data/derived/` (`aggregates.pkl`, `graph.pkl`, `redteam_full.csv`, `redteam_fit.csv`, `redteam_holdout.csv`, `redteam_profile.json`, `target_validation.json`).

**Skipped deliberately** (`ponytail:`): string→int ID encoding for nodes (SPEC suggests it). 17.7k string nodes + a few million edges fit RAM as plain strings; add encoding only if a real memory ceiling appears. NetworkX degrees are derived from the edge list rather than aggregated in SQL — same numbers, one fewer scan.

---

### Task 1: Project skeleton, config, deps, environment check

**Files:**
- Create: `config.yaml`, `.gitignore`, `requirements.txt`, `src/__init__.py`, `src/config.py`, `tests/__init__.py`

- [ ] **Step 1: Create the directory tree and package markers**

Run:
```bash
mkdir -p src scripts tests data/raw data/derived docs/superpowers/plans
: > src/__init__.py
: > tests/__init__.py
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
data/
__pycache__/
*.pyc
.pytest_cache/
*.duckdb
```

- [ ] **Step 3: Write `requirements.txt`**

```text
duckdb>=1.0
networkx>=3.4
numpy>=2.0
pandas>=2.2
scipy>=1.14
scikit-learn>=1.5
matplotlib>=3.9
pyyaml>=6.0
```

- [ ] **Step 4: Write `config.yaml`**

```yaml
seed: 42

paths:
  auth_gz: data/raw/auth.txt.gz
  redteam_gz: data/raw/redteam.txt.gz
  aggregates: data/derived/aggregates.pkl
  graph: data/derived/graph.pkl
  redteam_full: data/derived/redteam_full.csv
  redteam_fit: data/derived/redteam_fit.csv
  redteam_holdout: data/derived/redteam_holdout.csv
  profile: data/derived/redteam_profile.json
  target_validation: data/derived/target_validation.json

campaign_gap_seconds: 14400   # 4h (SETTLED). 1h fallback = 3600.

split:
  fit_fraction: 0.70          # earliest 70% of the timeline goes to fit

graph:
  server_in_degree_percentile: 90   # top-decile in-degree ...
  server_min_in_out_ratio: 3.0      # ... AND receives >> initiates
  target_keep_ratio: 2.0            # keep is_server proxy if targets' median in-degree >= 2x all-hosts

benchmark_rows: 10000000      # ~10M rows for the extrapolation
expected_total_rows: 1050000000
max_full_pass_hours: 4        # §5 fallback trigger
collection_days: 58

# Day-2 / Day-3 knobs, placed here now so the file is stable
alpha: 1.0
beta: 1.0
base_rate: 0.0001
```

- [ ] **Step 5: Write `src/config.py`**

```python
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 6: Verify the environment (DuckDB present and reads gzip)**

Run:
```bash
python -c "import duckdb, networkx, numpy, pandas, yaml, scipy, sklearn, matplotlib; print('deps ok', duckdb.__version__)"
```
Expected: `deps ok <version>`. If `duckdb` is missing (SPEC lists the others as verified), install it:
```bash
python -m pip install "duckdb>=1.0"
```

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt config.yaml src/__init__.py src/config.py tests/__init__.py
git commit -m "chore: day-1 skeleton, config, deps"
```

---

### Task 2: Test fixtures (tiny gzip auth + redteam with known aggregates)

**Files:**
- Create: `tests/conftest.py`

These six auth rows have hand-computed expected values reused across every parse test. Do not change them without updating the asserts.

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import gzip
import pytest

# 9-field auth: time,src_user,dst_user,src_computer,dst_computer,
#               auth_type,logon_type,auth_orientation,success
MINI_AUTH = (
    "1,U1@D1,U1@D1,C1,C2,Ntlm,Network,LogOn,Success\n"
    "2,U1@D1,U1@D1,C1,C2,Ntlm,Network,LogOn,Success\n"
    "3,U2@D1,U2@D1,C2,C3,Kerberos,Interactive,LogOn,Success\n"
    "4,U3@D1,U3@D1,C1,C3,Ntlm,Network,LogOn,Failure\n"
    "3601,U2@D1,U2@D1,C3,C4,?,Network,LogOff,Success\n"
    "7201,U4@D1,U4@D1,C4,C3,Kerberos,Network,LogOn,Success\n"
)

# 4-field redteam: time,user,src_computer,dst_computer
# First two rows exist in MINI_AUTH (join); third does not (unmatched).
MINI_REDTEAM = (
    "3,U2@D1,C2,C3\n"
    "7201,U4@D1,C4,C3\n"
    "99999,U9@D1,C9,C9\n"
)


def _write_gz(path, text):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)
    return path


@pytest.fixture
def mini_auth_path(tmp_path):
    return _write_gz(tmp_path / "mini_auth.txt.gz", MINI_AUTH)


@pytest.fixture
def mini_redteam_path(tmp_path):
    return _write_gz(tmp_path / "mini_redteam.txt.gz", MINI_REDTEAM)
```

- [ ] **Step 2: Sanity-check the fixture module imports**

Run:
```bash
python -m pytest tests/conftest.py -q
```
Expected: pytest collects 0 tests but reports no import/collection errors (`no tests ran`). This confirms the fixture module is syntactically valid.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: day-1 gzip fixtures with known aggregates"
```

---

### Task 3: DuckDB relations + the five aggregate queries

**Files:**
- Create: `src/parse.py`
- Test: `tests/test_parse.py`

Hand-computed expected values from the fixture:
- **edges:** (C1,C2)=2, (C2,C3)=1, (C1,C3)=1, (C3,C4)=1, (C4,C3)=1 → 5 edges.
- **user_host** (credential seen on a host = appears as src OR dst computer): (U1@D1,C1)=2, (U1@D1,C2)=2, (U2@D1,C2)=1, (U2@D1,C3)=2, (U2@D1,C4)=1, (U3@D1,C1)=1, (U3@D1,C3)=1, (U4@D1,C4)=1, (U4@D1,C3)=1.
- **hourly:** hour0=4, hour1=1, hour2=1.
- **marginals.auth_type:** Ntlm=3, Kerberos=2, `?`=1.

- [ ] **Step 1: Write the failing tests**

```python
import duckdb
import pytest
import src.parse as P


@pytest.fixture
def con():
    return duckdb.connect()


def _lookup(df, keys, val_col):
    return {tuple(r[k] for k in keys): r[val_col] for _, r in df.iterrows()}


def test_edge_counts(con, mini_auth_path):
    df = P.compute_edge_counts(con, mini_auth_path)
    d = _lookup(df, ["src_computer", "dst_computer"], "weight")
    assert d[("C1", "C2")] == 2
    assert d[("C2", "C3")] == 1
    assert d[("C4", "C3")] == 1
    assert len(df) == 5


def test_user_host_counts(con, mini_auth_path):
    df = P.compute_user_host_counts(con, mini_auth_path)
    d = _lookup(df, ["user", "computer"], "cnt")
    assert d[("U1@D1", "C1")] == 2
    assert d[("U1@D1", "C2")] == 2
    assert d[("U2@D1", "C3")] == 2   # row3 dst + row5 src
    assert d[("U4@D1", "C3")] == 1


def test_hourly_volume(con, mini_auth_path):
    df = P.compute_hourly_volume(con, mini_auth_path)
    d = _lookup(df, ["hour"], "cnt")
    assert d[(0,)] == 4 and d[(1,)] == 1 and d[(2,)] == 1


def test_marginals(con, mini_auth_path):
    m = P.compute_marginals(con, mini_auth_path)
    at = _lookup(m["auth_type"], ["value"], "cnt")
    assert at[("Ntlm",)] == 3 and at[("Kerberos",)] == 2 and at[("?",)] == 1
    assert set(m.keys()) == {"auth_type", "logon_type", "auth_orientation", "success"}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_parse.py -q`
Expected: FAIL — `AttributeError: module 'src.parse' has no attribute 'compute_edge_counts'`.

- [ ] **Step 3: Write `src/parse.py` (relations + aggregates)**

```python
import pickle

import duckdb
import numpy as np
import pandas as pd
import networkx as nx

AUTH_COLS = ("time", "src_user", "dst_user", "src_computer", "dst_computer",
             "auth_type", "logon_type", "auth_orientation", "success")


def _auth_rel(auth_path):
    # ponytail: inline the trusted config path (single-quote escaped) rather than
    # bind params — read_csv columns/compression are literals, and inlining keeps
    # one relation-builder reused by every query below.
    p = str(auth_path).replace("\\", "/").replace("'", "''")
    return (f"read_csv('{p}', header=false, delim=',', compression='gzip', "
            "columns={'time':'BIGINT','src_user':'VARCHAR','dst_user':'VARCHAR',"
            "'src_computer':'VARCHAR','dst_computer':'VARCHAR','auth_type':'VARCHAR',"
            "'logon_type':'VARCHAR','auth_orientation':'VARCHAR','success':'VARCHAR'})")


def _redteam_rel(rt_path):
    p = str(rt_path).replace("\\", "/").replace("'", "''")
    return (f"read_csv('{p}', header=false, delim=',', compression='gzip', "
            "columns={'time':'BIGINT','user':'VARCHAR','src_computer':'VARCHAR',"
            "'dst_computer':'VARCHAR'})")


def compute_edge_counts(con, auth_path):
    return con.execute(
        f"SELECT src_computer, dst_computer, COUNT(*) AS weight "
        f"FROM {_auth_rel(auth_path)} GROUP BY src_computer, dst_computer"
    ).df()


def compute_user_host_counts(con, auth_path):
    # A credential is "seen on" a host if it appears as its source OR destination.
    # UNNEST a 2-element list per row => both orientations in a single file scan.
    return con.execute(
        f"SELECT user, computer, COUNT(*) AS cnt FROM ("
        f"  SELECT src_user AS user, "
        f"         UNNEST([src_computer, dst_computer]) AS computer "
        f"  FROM {_auth_rel(auth_path)}) GROUP BY user, computer"
    ).df()


def compute_hourly_volume(con, auth_path):
    return con.execute(
        f"SELECT (time // 3600) % 24 AS hour, COUNT(*) AS cnt "
        f"FROM {_auth_rel(auth_path)} GROUP BY 1 ORDER BY 1"
    ).df()


def compute_marginals(con, auth_path):
    rel = _auth_rel(auth_path)
    out = {}
    for col in ("auth_type", "logon_type", "auth_orientation", "success"):
        out[col] = con.execute(
            f"SELECT {col} AS value, COUNT(*) AS cnt FROM {rel} GROUP BY 1"
        ).df()
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_parse.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/parse.py tests/test_parse.py
git commit -m "feat: duckdb streaming aggregates (edges, user-host, hourly, marginals)"
```

---

### Task 4: Red-team full-record recovery join

**Files:**
- Modify: `src/parse.py`
- Test: `tests/test_parse.py`

Fixture: 3 red-team rows, 2 join to `auth`, 1 (`99999,U9@D1,C9,C9`) does not → `full` has 2 rows, `unmatched == 1`.

- [ ] **Step 1: Add the failing test**

```python
def test_recover_redteam(con, mini_auth_path, mini_redteam_path):
    full, unmatched = P.recover_redteam(con, mini_auth_path, mini_redteam_path)
    assert unmatched == 1
    assert len(full) == 2
    assert list(full.columns) == list(P.AUTH_COLS)
    row = full[full["time"] == 3].iloc[0]
    assert row["auth_type"] == "Kerberos" and row["dst_computer"] == "C3"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_parse.py::test_recover_redteam -q`
Expected: FAIL — `has no attribute 'recover_redteam'`.

- [ ] **Step 3: Add `recover_redteam` to `src/parse.py`**

```python
def recover_redteam(con, auth_path, rt_path):
    """LEFT JOIN redteam onto auth on (time,user,src,dst); recover the 9-field
    record. Returns (full_df in AUTH_COLS order, n_unmatched)."""
    q = f"""
      SELECT r.time, r.user AS src_user, a.dst_user, r.src_computer, r.dst_computer,
             a.auth_type, a.logon_type, a.auth_orientation, a.success
      FROM {_redteam_rel(rt_path)} r
      LEFT JOIN {_auth_rel(auth_path)} a
        ON a.time = r.time AND a.src_user = r.user
       AND a.src_computer = r.src_computer AND a.dst_computer = r.dst_computer
    """
    df = con.execute(q).df()
    df = df.drop_duplicates(subset=["time", "src_user", "src_computer", "dst_computer"])
    unmatched = int(df["auth_type"].isna().sum())
    full = df[df["auth_type"].notna()].copy()
    full = full[list(AUTH_COLS)].reset_index(drop=True)
    return full, unmatched
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_parse.py::test_recover_redteam -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/parse.py tests/test_parse.py
git commit -m "feat: recover 9-field redteam records via left join"
```

---

### Task 5: Build the graph with derived `is_server`

**Files:**
- Modify: `src/parse.py`
- Test: `tests/test_parse.py`

Fixture degrees (unweighted DiGraph): C3 in=3/out=1/ratio=1.5 (most server-like), C1 in=0/out=2, C2 in=1/out=1, C4 in=1/out=1. The test passes its own thresholds (`percentile=75, min_ratio=1.0`) so C3 is the only server; production `config.yaml` uses 90 / 3.0.

- [ ] **Step 1: Add the failing test**

```python
import networkx as nx


def test_build_graph_degrees_and_is_server(con, mini_auth_path):
    edges = P.compute_edge_counts(con, mini_auth_path)
    g = P.build_graph(edges, server_in_degree_percentile=75, server_min_in_out_ratio=1.0)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 5
    assert g.nodes["C3"]["in_degree"] == 3
    assert g.nodes["C3"]["out_degree"] == 1
    assert abs(g.nodes["C3"]["in_out_ratio"] - 1.5) < 1e-9
    assert g["C1"]["C2"]["weight"] == 2
    assert g.nodes["C3"]["is_server"] is True
    assert g.nodes["C1"]["is_server"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_parse.py::test_build_graph_degrees_and_is_server -q`
Expected: FAIL — `has no attribute 'build_graph'`.

- [ ] **Step 3: Add graph functions to `src/parse.py`**

```python
def build_graph(edge_counts, server_in_degree_percentile=90, server_min_in_out_ratio=3.0):
    """DiGraph from edge counts. Nodes carry in/out degree, in_out_ratio,
    and a derived is_server (top in-degree percentile AND receives >> initiates)."""
    g = nx.DiGraph()
    for r in edge_counts.itertuples(index=False):
        g.add_edge(r.src_computer, r.dst_computer, weight=int(r.weight))

    for n in g.nodes:
        indeg, outdeg = g.in_degree(n), g.out_degree(n)
        g.nodes[n]["in_degree"] = int(indeg)
        g.nodes[n]["out_degree"] = int(outdeg)
        g.nodes[n]["in_out_ratio"] = indeg / (outdeg + 1)

    indeg_arr = np.array([g.nodes[n]["in_degree"] for n in g.nodes], dtype=float)
    thr = np.percentile(indeg_arr, server_in_degree_percentile)
    for n in g.nodes:
        g.nodes[n]["is_server"] = bool(
            g.nodes[n]["in_degree"] >= thr
            and g.nodes[n]["in_out_ratio"] >= server_min_in_out_ratio
        )
    return g


def save_graph(g, path):
    with open(path, "wb") as f:
        pickle.dump(g, f)


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_parse.py::test_build_graph_degrees_and_is_server -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/parse.py tests/test_parse.py
git commit -m "feat: build DiGraph with degrees and derived is_server"
```

---

### Task 6: Target-attractiveness Day-1 validation (the keep/drop-proxy gate)

**Files:**
- Modify: `src/parse.py`
- Test: `tests/test_parse.py`

SPEC §3 Day-1 validation: compare the in-degree distribution of the real red-team **targets** against all hosts. If targets are clearly high-in-degree, keep the `is_server` proxy; if muddy, drop it and weight candidates by similarity to the empirical target-profile on Day 2. Fixture target = {C3} (in-degree 3); all-hosts median in-degree = 1.0 → ratio 3.0 ≥ 2.0 → keep.

- [ ] **Step 1: Add the failing test**

```python
def test_validate_target_attractiveness(con, mini_auth_path, mini_redteam_path):
    edges = P.compute_edge_counts(con, mini_auth_path)
    g = P.build_graph(edges, server_in_degree_percentile=75, server_min_in_out_ratio=1.0)
    full, _ = P.recover_redteam(con, mini_auth_path, mini_redteam_path)
    res = P.validate_target_attractiveness(g, full, keep_ratio=2.0)
    assert res["n_targets_in_graph"] == 1          # C3 (C9 not in graph)
    assert res["target_indegree_median"] == 3.0
    assert res["all_indegree_median"] == 1.0
    assert res["separation_ratio"] == 3.0
    assert res["keep_server_proxy"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_parse.py::test_validate_target_attractiveness -q`
Expected: FAIL — `has no attribute 'validate_target_attractiveness'`.

- [ ] **Step 3: Add `validate_target_attractiveness` to `src/parse.py`**

```python
def validate_target_attractiveness(graph, redteam_full, keep_ratio=2.0):
    """Structure-only check: are red-team target hosts high-in-degree vs all hosts?
    Returns a report dict and the keep/drop decision for the is_server proxy."""
    all_indeg = np.array([graph.nodes[n]["in_degree"] for n in graph.nodes], dtype=float)
    targets = [t for t in redteam_full["dst_computer"].unique() if t in graph]
    tgt_indeg = np.array([graph.nodes[t]["in_degree"] for t in targets], dtype=float)

    tmed = float(np.median(tgt_indeg)) if len(tgt_indeg) else 0.0
    amed = float(np.median(all_indeg)) if len(all_indeg) else 0.0
    ratio = (tmed / amed) if amed > 0 else float("inf")

    return {
        "n_targets_total": int(redteam_full["dst_computer"].nunique()),
        "n_targets_in_graph": len(targets),
        "target_indegree_median": tmed,
        "all_indegree_median": amed,
        "separation_ratio": ratio,
        "keep_server_proxy": bool(ratio >= keep_ratio),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_parse.py::test_validate_target_attractiveness -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/parse.py tests/test_parse.py
git commit -m "feat: day-1 target-attractiveness validation gate"
```

---

### Task 7: Sessionize red-team events into campaigns

**Files:**
- Create: `src/profile.py`
- Test: `tests/test_profile.py`

`campaign_gap = 4h`: an inter-event gap greater than the threshold starts a new campaign. Test uses three clearly separated bursts.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
import networkx as nx
import src.profile as PR


def _rt(rows):
    return pd.DataFrame(rows, columns=[
        "time", "src_user", "dst_user", "src_computer", "dst_computer",
        "auth_type", "logon_type", "auth_orientation", "success"])


def test_sessionize_gap():
    df = _rt([
        [0, "U1@D", "U1@D", "C1", "C2", "Ntlm", "Network", "LogOn", "Success"],
        [60, "U1@D", "U1@D", "C2", "C3", "Ntlm", "Network", "LogOn", "Success"],
        [100000, "U1@D", "U1@D", "C3", "C4", "Ntlm", "Network", "LogOn", "Success"],
        [100060, "U1@D", "U1@D", "C4", "C5", "Ntlm", "Network", "LogOn", "Success"],
        [500000, "U1@D", "U1@D", "C5", "C6", "Ntlm", "Network", "LogOn", "Success"],
        [500060, "U1@D", "U1@D", "C6", "C7", "Ntlm", "Network", "LogOn", "Success"],
    ])
    out = PR.sessionize(df, gap_seconds=14400)
    assert out["campaign_id"].tolist() == [0, 0, 1, 1, 2, 2]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_profile.py::test_sessionize_gap -q`
Expected: FAIL — `No module named 'src.profile'`.

- [ ] **Step 3: Write `src/profile.py` (sessionize)**

```python
import json

import numpy as np
import pandas as pd


def sessionize(df, gap_seconds):
    """Global time-gap sessionization: a gap > gap_seconds starts a new campaign."""
    df = df.sort_values("time").reset_index(drop=True)
    df["campaign_id"] = (df["time"].diff() > gap_seconds).cumsum().astype(int)
    return df
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_profile.py::test_sessionize_gap -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/profile.py tests/test_profile.py
git commit -m "feat: sessionize redteam events into campaigns (4h gap)"
```

---

### Task 8: Freeze the whole-campaign 70%-timeline split

**Files:**
- Modify: `src/profile.py`
- Test: `tests/test_profile.py`

Whole-campaign assignment: cut at the inter-campaign gap nearest the 70%-of-timeline mark; a campaign is never sliced. With the three campaigns above (`t0=0, t1=500060`, cut≈350042), the boundary between campaign 1 and 2 (gap `[100060, 500000]`) contains the cut → fit = campaigns {0,1} (4 rows), holdout = {2} (2 rows).

- [ ] **Step 1: Add the failing test**

```python
def test_freeze_split_whole_campaign():
    df = _rt([
        [0, "U1@D", "U1@D", "C1", "C2", "Ntlm", "Network", "LogOn", "Success"],
        [60, "U1@D", "U1@D", "C2", "C3", "Ntlm", "Network", "LogOn", "Success"],
        [100000, "U1@D", "U1@D", "C3", "C4", "Ntlm", "Network", "LogOn", "Success"],
        [100060, "U1@D", "U1@D", "C4", "C5", "Ntlm", "Network", "LogOn", "Success"],
        [500000, "U1@D", "U1@D", "C5", "C6", "Ntlm", "Network", "LogOn", "Success"],
        [500060, "U1@D", "U1@D", "C6", "C7", "Ntlm", "Network", "LogOn", "Success"],
    ])
    s = PR.sessionize(df, gap_seconds=14400)
    fit, holdout = PR.freeze_split(s, fit_fraction=0.70)
    assert sorted(fit["campaign_id"].unique().tolist()) == [0, 1]
    assert holdout["campaign_id"].unique().tolist() == [2]
    assert len(fit) == 4 and len(holdout) == 2
    # no campaign straddles the boundary
    assert set(fit["campaign_id"]).isdisjoint(set(holdout["campaign_id"]))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_profile.py::test_freeze_split_whole_campaign -q`
Expected: FAIL — `has no attribute 'freeze_split'`.

- [ ] **Step 3: Add `freeze_split` to `src/profile.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_profile.py::test_freeze_split_whole_campaign -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/profile.py tests/test_profile.py
git commit -m "feat: whole-campaign 70%-timeline split freeze"
```

---

### Task 9: Build `redteam_profile.json` from the fit split

**Files:**
- Modify: `src/profile.py`
- Test: `tests/test_profile.py`

Captures every distribution SPEC §3 Day-1 t6 requires: chain length, inter-hop Δt, seed-host and target-host characteristics, the conditional attribute table (keyed as SPEC §3.1: `dst_is_server | hop_bucket | auth_orientation → (auth_type,logon_type,success)`), and the credential-reuse pattern.

- [ ] **Step 1: Add the failing test**

```python
def test_build_profile_keys():
    fit = _rt([
        [0, "U1@D", "U1@D", "C1", "C2", "Ntlm", "Network", "LogOn", "Success"],
        [60, "U2@D", "U2@D", "C2", "C3", "Kerberos", "Network", "LogOn", "Success"],
        [120, "U1@D", "U1@D", "C3", "C4", "Ntlm", "Network", "LogOn", "Success"],
        [100000, "U1@D", "U1@D", "C1", "C3", "Ntlm", "Network", "LogOn", "Success"],
    ])
    fit = PR.sessionize(fit, gap_seconds=14400)  # -> campaigns {0:3 events, 1:1 event}
    g = nx.DiGraph()
    for u, v in [("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C1", "C3")]:
        g.add_edge(u, v)
    for n in g.nodes:
        g.nodes[n]["in_degree"] = g.in_degree(n)
        g.nodes[n]["out_degree"] = g.out_degree(n)
        g.nodes[n]["in_out_ratio"] = g.in_degree(n) / (g.out_degree(n) + 1)
        g.nodes[n]["is_server"] = g.in_degree(n) >= 2

    prof = PR.build_profile(fit, g)
    assert sorted(prof["chain_length"]["values"]) == [1, 3]
    assert prof["chain_length"]["median"] == 2.0
    assert len(prof["inter_hop_dt"]["values"]) == 2         # two positive gaps in campaign 0
    assert all(x > 0 for x in prof["inter_hop_dt"]["values"])
    for key in ("seed_hosts", "target_indegree", "conditional_attributes",
                "credential_reuse"):
        assert key in prof
    assert prof["credential_reuse"][0]["n_distinct_credentials"] == 2  # U1,U2 in campaign 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_profile.py::test_build_profile_keys -q`
Expected: FAIL — `has no attribute 'build_profile'`.

- [ ] **Step 3: Add profiling functions to `src/profile.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_profile.py -q`
Expected: PASS (all profile tests).

- [ ] **Step 5: Commit**

```bash
git add src/profile.py tests/test_profile.py
git commit -m "feat: build redteam_profile.json from fit split"
```

---

### Task 10: Download the data (manual action + checksums)

**Files:**
- Create: `data/raw/auth.txt.gz`, `data/raw/redteam.txt.gz`, `data/raw/CHECKSUMS.txt` (all gitignored)

SPEC §5 risk: if the download is unavailable/throttled, escalate within 2 hours — it blocks everything. Do **not** decompress to disk.

- [ ] **Step 1: Download both files** (run interactively in the session prompt with `!` so output is captured; downloads are large — ~11–12 GB for auth)

```
! curl.exe -L -o "data/raw/redteam.txt.gz" https://csr.lanl.gov/data/cyber1/redteam.txt.gz
! curl.exe -L -o "data/raw/auth.txt.gz"    https://csr.lanl.gov/data/cyber1/auth.txt.gz
```
If the portal requires a click-through/agreement, download in the browser and drop the files into `data/raw/`.

- [ ] **Step 2: Record checksums**

Run (PowerShell):
```powershell
Get-FileHash data/raw/auth.txt.gz,data/raw/redteam.txt.gz -Algorithm SHA256 |
  ForEach-Object { "$($_.Hash)  $($_.Path)" } | Out-File -Encoding utf8 data/raw/CHECKSUMS.txt
Get-Content data/raw/CHECKSUMS.txt
```
Expected: two SHA256 lines. Record the actual on-disk sizes here too (SPEC §2.1: verify, do not trust ~11–12 GB / 749-row figures).

- [ ] **Step 3: Confirm DuckDB reads the gzip without decompressing**

Run:
```bash
python -c "import duckdb, src.parse as P; c=duckdb.connect(); print(c.execute(f'SELECT * FROM {P._auth_rel(\"data/raw/auth.txt.gz\")} LIMIT 3').df())"
```
Expected: 3 well-formed 9-column rows. Missing values show as `?` (a real category — never dropped).

- [ ] **Step 4: Commit the checksum record only** (data is gitignored)

```bash
git add -f data/raw/CHECKSUMS.txt
git commit -m "chore: record source data checksums"
```

---

### Task 11: Benchmark, then run the full pass and check exit criteria

**Files:**
- Create: `scripts/benchmark.py`, `scripts/run_day1.py`

SPEC §3 Day-1 t2 + §5: benchmark on the first ~10M rows and extrapolate **before** committing to the full pass. If the extrapolation exceeds `max_full_pass_hours`, fall back to a contiguous time-slice covering the fit-split red-team window and document the reduced coverage.

- [ ] **Step 1: Write `scripts/benchmark.py`**

```python
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
```

- [ ] **Step 2: Run the benchmark**

Run: `python -m scripts.benchmark`
Expected: a timing line and an hours estimate. Proceed only if under `max_full_pass_hours`; otherwise apply the printed §5 fallback before Step 4.

- [ ] **Step 3: Write `scripts/run_day1.py` (orchestrator + exit-criteria checklist)**

```python
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
    con = duckdb.connect()

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
```

- [ ] **Step 4: Run the full pass on real data**

Run: `python -m scripts.run_day1`
Expected: six progress lines then the exit-criteria checklist, ideally all `[x]`. Node count near ~17,684; `unmatched` small and explained; split roughly fit ≈ 690 / holdout ≈ 59 events over ~19 campaigns (SPEC §2.3 measured reality — the ~8% holdout mass is intrinsic, do not rebalance).

- [ ] **Step 5: Confirm the whole test suite still passes**

Run: `python -m pytest -q`
Expected: PASS (all parse + profile tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark.py scripts/run_day1.py
git commit -m "feat: day-1 benchmark + full-pass orchestrator with exit checks"
```

---

## Day-1 Exit Criteria (SPEC §3 DAY 1) — verify before the phase-gate

- [ ] Graph loads from disk; node/edge counts printed and sanity-checked against SPEC §2.2 (~17,684 computers).
- [ ] All red-team events joined to full 9-field records; any that fail to join are counted and explained (`unmatched` in Task 11).
- [ ] Split files (`redteam_fit.csv`, `redteam_holdout.csv`) written and immutable (read-only).
- [ ] `redteam_profile.json` contains every listed distribution.

**Do not start Day 2.** SPEC §3.0 mandates a grilling checkpoint (review Day-1 state, revise Day-2 knobs, explicit go/no-go) before the walk generator.

---

## Self-Review

- **Spec coverage (SPEC §3 DAY 1 t1–t6):** t1 download+checksums → Task 10; t2 streaming aggregates → Tasks 3 (+ benchmark Task 11); t3 graph + `is_server` + target-attractiveness → Tasks 5, 6; t4 redteam recovery join → Task 4; t5 freeze split → Tasks 7, 8; t6 profile → Task 9. All exit criteria wired into `run_day1.py`.
- **Placeholder scan:** every code step contains full runnable code; no TBD/TODO/"handle edge cases".
- **Type consistency:** `AUTH_COLS` is the single source of the 9-field order (Task 3), reused by `recover_redteam` (Task 4) and consumed by profile/sessionize which key off `campaign_id`, `hop_index`, `src_user`, `dst_computer` consistently. `build_graph` signature (percentile, min_ratio) is identical in tests and `run_day1.py`. `validate_target_attractiveness` return keys match test asserts and orchestrator reads.
- **Known ceiling:** DuckDB re-scans the gzip once per aggregate query (≈6 scans). Benchmark Task 11 measures this against `max_full_pass_hours` and prints the §5 time-slice fallback if it blows the budget — the upgrade path if throughput matters is materializing the projected columns once.
