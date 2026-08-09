# Day 3 — Attribute + Time + Writer + Benign + Mixed Corpus — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Day-2 synthetic fan-out campaigns into a **data product**: LANL-format `synth_auth.csv` + row-aligned `synth_labels.csv`, plus **windowed real-benign samples** and a **mixed corpus** (synthetic malicious injected into real benign at configurable base rates) — the inputs Day-4 augmentation evaluation consumes.

**Architecture:** Four layers on top of `data/derived/synth_campaigns.pkl`. (1) A **creds-per-campaign fix** in `walker.py` so credential-reuse intensity matches the fit split. (2) A **writer** that attaches the constant attribute tuple `(NTLM, Network, LogOn, Success)`, places each campaign on an absolute timeline (start sampled ∝ real `hourly` volume, `dt_offset` → absolute `time`, kept inside the collection window), and emits exact 9-field LANL rows + labels. (3) A **benign extractor** that streams `auth.txt.gz` once via the Day-1 DuckDB tooling and pulls a random benign sample per split window (fit-window → train negatives, holdout-window → test negatives), dropping rows on known red-team edges. (4) A **mixed-corpus** builder that interleaves malicious rows into benign at a sweep of base rates. Attributes are constant by measurement (SPEC v1.2) — no conditional machinery.

**Tech Stack:** Python 3.11, DuckDB (reuse `src/parse.open_con` + `_auth_rel`), NumPy, Pandas, PyYAML, pytest. Reuses Day-1/2 artifacts: `data/derived/graph.pkl`, `aggregates.pkl` (`hourly`), `redteam_fit.csv`, `redteam_full.csv`, `synth_campaigns.pkl`, `config.py`, `walker.py`, `parse.py`.

**Scope:** Day 3 only (data production). Day-4 augmentation/eval (scarcity curve, TSTR, baselines, ablation, memorization metric) is a **separate plan** — do not build it here. The SPEC §3.0 phase-gate grilling checkpoint sits between Day 3 and Day 4.

**Grounded facts (verified on real artifacts — trust, don't re-derive):**
- LANL field order (`src/parse.py` `AUTH_COLS`): `time, src_user, dst_user, src_computer, dst_computer, auth_type, logon_type, auth_orientation, success`.
- Constant attribute tuple: `auth_type=NTLM, logon_type=Network, auth_orientation=LogOn, success=Success`; `dst_user == src_user` (100% in fit).
- Synth campaign = `list[(int dt_offset, str user, str src_computer, str dst_computer)]`, offsets strictly increasing from 0.
- Split windows: **fit** `time ∈ [150885, 1758226]`, **holdout** `time ∈ [1847858, 2557047]`. Max observed red-team time `2557047` (~29.6 d); collection window `58 d = 5011200 s`.
- `aggregates.pkl["hourly"]` = DataFrame cols `[hour, cnt]`, 24 rows, peaks hours 7–15 (matches the real red-team 6–17 business-hours footprint).
- `synth_campaigns.pkl` currently holds 10,000 campaigns (regenerated in Task 7 after the creds fix).

**Decisions this plan encodes (from the Day-3 grilling, `docs/superpowers/phase-gate-day1-to-day2.md`) — do not relitigate:**
- Purpose = **augmentation** (Day-4 concern; here we only produce the data it needs).
- **Fix creds-per-campaign** (gap #2). **Keep foothold pool broad** (gap #3 — novelty). **Defer** multi-origin (#5) + cred→target proxy (#1) to limitations.
- Benign **windowed to the split**, random draw ~1–2M rows/window, drop known-red-team edges, document residual label noise.
- Base-rate **sweep 1:100 / 1:1000 / 1:10000**.
- Time start ∝ benign `hourly` volume; verify footprint ≈ 6–17.
- `hop_index` in labels = **within-campaign emission order** (fan-out has no hops).

---

## File Structure

| File | Responsibility |
|---|---|
| `config.yaml` | Add a `day3:` block (output paths, benign sample sizes, split window bounds, base-rate sweep list). |
| `src/walker.py` | **Modify** `generate_campaign`: after harvest, restrict the compromised set to `m` creds where `m` ~ `dists["creds_per_campaign"]` (gap #2 fix). No signature change. |
| `src/writer.py` | Constant-attribute emission, start-time sampling ∝ hourly, `dt_offset`→absolute placement, 9-field LANL row + label construction, CSV writers. |
| `src/benign.py` | DuckDB windowed random benign sampler over `auth.txt.gz`, red-team-edge filter. |
| `src/mixed.py` | Interleave malicious rows into benign at a base rate → mixed corpus + aligned labels. |
| `scripts/run_day3.py` | Orchestrate: (re)generate campaigns with the cred fix, place+write synth rows+labels, extract both benign windows, build mixed corpora at the sweep rates, assert exit criteria. |
| `tests/test_walker.py` | **Add** a creds-per-campaign cap test. |
| `tests/test_writer.py` | Unit tests for writer functions. |
| `tests/test_benign.py` | Unit tests for the benign sampler against a tiny in-memory DuckDB table. |
| `tests/test_mixed.py` | Unit tests for the mixed-corpus builder. |

**Data produced at runtime:** `data/derived/synth_auth.csv`, `synth_labels.csv`, `benign_fit.csv`, `benign_holdout.csv`, `mixed_1e2.csv` / `mixed_1e3.csv` / `mixed_1e4.csv` (+ matching `*_labels.csv`). All git-ignored like other `data/derived/*`.

**Skipped deliberately** (`ponytail:`): no conditional attribute table (constant by measurement); no benign feature engineering (Day-4); no multi-origin or cred-proxy fix (deferred to limitations).

---

### Task 1: Add the `day3` config block

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Append the `day3` block to `config.yaml`**

Add at the end (leave everything else as-is):

```yaml
# Day-3 writer / benign / mixed-corpus knobs
day3:
  synth_auth: data/derived/synth_auth.csv
  synth_labels: data/derived/synth_labels.csv
  benign_fit: data/derived/benign_fit.csv
  benign_holdout: data/derived/benign_holdout.csv
  fit_window: [0, 1758226]          # benign train negatives: time <= fit max
  holdout_window: [1847858, 2557047] # benign test negatives: holdout span
  benign_n: 1500000                  # random benign rows per window
  collection_seconds: 5011200        # 58-day window; campaign start kept inside it
  base_rates:                        # malicious:benign ratios for the mixed-corpus sweep
    - {name: "1e2", rate: 0.01}
    - {name: "1e3", rate: 0.001}
    - {name: "1e4", rate: 0.0001}
```

- [ ] **Step 2: Verify it parses**

Run:
```bash
python -c "from src.config import load_config; c=load_config(); d=c['day3']; print(d['benign_n'], d['fit_window'], [b['name'] for b in d['base_rates']])"
```
Expected: `1500000 [0, 1758226] ['1e2', '1e3', '1e4']`

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "chore: day-3 writer/benign/mixed config knobs"
```

---

### Task 2: Enforce creds-per-campaign in the generator (gap #2 fix)

**Files:**
- Modify: `src/walker.py` (`generate_campaign`)
- Test: `tests/test_walker.py`

The generator currently harvests the whole foothold credential set, so the number of distinct credentials a campaign reuses is emergent. Fix: after harvest, draw a target count `m` from the real `creds_per_campaign` distribution (already in `dists`) and restrict the compromised set to `m` credentials. This ties reuse *intensity* to the fit split without changing the function signature.

- [ ] **Step 1: Add the failing test**

```python
def test_generate_campaign_restricts_to_sampled_cred_count():
    g = _graph_with_targets()
    # foothold F has 5 creds available, but creds_per_campaign says a campaign uses 2
    host_users = {"F": {f"U{i}@D" for i in range(5)},
                  "T1": {"U0@D"}, "T2": {"U1@D"}, "T3": {"U2@D"}}
    dists = {"breadth": [3], "creds_per_campaign": [2], "inter_event_dt": [60]}
    cands, cp = W.foothold_candidates(g, min_out_degree=1)
    events, _ = W.generate_campaign(
        g, host_users, dists, cands, cp, _rng(),
        alpha=1.0, beta=1.0, credential_bonus=2.0, max_creds=60)
    assert events is not None
    assert len({user for _, user, _, _ in events}) <= 2   # never more than sampled m
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_walker.py::test_generate_campaign_restricts_to_sampled_cred_count -v`
Expected: FAIL — up to 3 distinct users appear (the fix isn't in yet).

- [ ] **Step 3: Apply the fix in `generate_campaign`**

In `src/walker.py`, find these lines in `generate_campaign`:

```python
    foothold = pick_foothold(cands, cp, rng)
    compromised = harvest_credentials(foothold, host_users, max_creds, rng)
    targets = list(graph.successors(foothold))
    if not compromised or not targets:
        return None, False
```

Replace with (add the `m`-restriction after the guard):

```python
    foothold = pick_foothold(cands, cp, rng)
    compromised = harvest_credentials(foothold, host_users, max_creds, rng)
    targets = list(graph.successors(foothold))
    if not compromised or not targets:
        return None, False

    # gap #2 fix: reuse-intensity matches the fit split — restrict to m sampled creds
    m = int(rng.choice(dists["creds_per_campaign"]))
    comp_list = list(compromised)
    if 0 < m < len(comp_list):
        idx = rng.choice(len(comp_list), size=m, replace=False)
        compromised = {comp_list[i] for i in idx}
```

- [ ] **Step 4: Run the whole walker suite**

Run: `python -m pytest tests/test_walker.py -q`
Expected: PASS (all prior Day-2 tests + the new one — 20 tests). The Day-2 tests are unaffected because their `compromised` sets are already ≤ their `m`.

- [ ] **Step 5: Commit**

```bash
git add src/walker.py tests/test_walker.py
git commit -m "fix: enforce creds-per-campaign reuse intensity in fan-out generator"
```

---

### Task 3: Time placement — start-time sampler + absolute placement

**Files:**
- Create: `src/writer.py`
- Test: `tests/test_writer.py`

`sample_start_time` draws a campaign start weighted by the real `hourly` volume (hour-of-day ∝ `cnt`), on a uniformly chosen day, kept far enough inside the collection window that `start + max_offset` never overflows. `place_campaign` shifts a campaign's relative offsets to absolute `time` values.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
import pytest
import src.writer as Wr


def _rng():
    return np.random.default_rng(7)


def _hourly():
    # volume only in hour 9 -> every sampled start must land in hour 9
    return pd.DataFrame({"hour": list(range(24)),
                         "cnt": [0]*9 + [1000] + [0]*14})


def test_sample_start_time_respects_hourly_and_window():
    rng = _rng()
    for _ in range(50):
        s = Wr.sample_start_time(_hourly(), rng, max_offset=100, collection_seconds=5011200)
        assert (s // 3600) % 24 == 9            # only hour 9 has volume
        assert 0 <= s
        assert s + 100 <= 5011200               # stays inside the window


def test_place_campaign_shifts_offsets_to_absolute():
    events = [(0, "U1@D", "F", "T1"), (60, "U1@D", "F", "T2"), (150, "U2@D", "F", "T3")]
    placed = Wr.place_campaign(events, start=1000)
    assert [p[0] for p in placed] == [1000, 1060, 1150]
    assert [p[1:] for p in placed] == [e[1:] for e in events]   # identities unchanged
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_writer.py -q`
Expected: FAIL — `module 'src.writer' has no attribute 'sample_start_time'`.

- [ ] **Step 3: Write `src/writer.py` (time layer)**

```python
"""Day-3 writer: constant attributes, time placement, LANL-format emission,
labels. Attributes are constant by measurement (SPEC v1.2): every synthetic
malicious auth is (NTLM, Network, LogOn, Success) with dst_user == src_user.
"""
import csv

import numpy as np
import pandas as pd

# Constant attribute tuple (SPEC v1.2 — fit split is 100% these values).
AUTH_TYPE = "NTLM"
LOGON_TYPE = "Network"
AUTH_ORIENTATION = "LogOn"
SUCCESS = "Success"

# Exact LANL field order (matches src/parse.AUTH_COLS).
AUTH_FIELDS = ["time", "src_user", "dst_user", "src_computer", "dst_computer",
               "auth_type", "logon_type", "auth_orientation", "success"]
LABEL_FIELDS = ["row_id", "is_malicious", "campaign_id", "hop_index"]

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


def sample_start_time(hourly_df, rng, max_offset, collection_seconds):
    """Absolute campaign start (seconds), hour-of-day ∝ real hourly volume,
    day uniform, kept so start + max_offset stays inside the collection window."""
    hours = hourly_df["hour"].to_numpy()
    p = hourly_df["cnt"].to_numpy(dtype=float)
    p = p / p.sum()
    latest_start = max(collection_seconds - max_offset, SECONDS_PER_DAY)
    n_days = max(latest_start // SECONDS_PER_DAY, 1)
    while True:
        day = int(rng.integers(0, n_days))
        hour = int(rng.choice(hours, p=p))
        sec = int(rng.integers(0, SECONDS_PER_HOUR))
        start = day * SECONDS_PER_DAY + hour * SECONDS_PER_HOUR + sec
        if start + max_offset <= collection_seconds:
            return start


def place_campaign(events, start):
    """Shift relative (dt_offset, user, src, dst) events to absolute time."""
    return [(int(start + off), user, src, dst) for off, user, src, dst in events]
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_writer.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/writer.py tests/test_writer.py
git commit -m "feat: day-3 time placement (hourly-weighted start + absolute offsets)"
```

---

### Task 4: LANL row + label emission

**Files:**
- Modify: `src/writer.py`
- Test: `tests/test_writer.py`

`campaign_to_rows` turns one placed campaign into 9-field LANL row dicts (constant attributes, `dst_user == src_user`) plus per-event label dicts (`is_malicious=1`, `campaign_id`, `hop_index` = within-campaign emission order). `write_auth_csv` / `write_labels_csv` emit headerless auth rows (LANL has no header) and a headed labels file, row-aligned by `row_id`.

- [ ] **Step 1: Add the failing tests**

```python
def test_campaign_to_rows_constant_attrs_and_labels():
    placed = [(1000, "U1@D", "F", "T1"), (1060, "U2@D", "F", "T2")]
    rows, labels = Wr.campaign_to_rows(placed, campaign_id=5, row_id_start=10)
    assert len(rows) == 2 and len(labels) == 2
    r0 = rows[0]
    assert r0["time"] == 1000 and r0["src_user"] == "U1@D" and r0["dst_user"] == "U1@D"
    assert r0["src_computer"] == "F" and r0["dst_computer"] == "T1"
    assert (r0["auth_type"], r0["logon_type"], r0["auth_orientation"], r0["success"]) \
        == ("NTLM", "Network", "LogOn", "Success")
    assert labels[0] == {"row_id": 10, "is_malicious": 1, "campaign_id": 5, "hop_index": 0}
    assert labels[1]["row_id"] == 11 and labels[1]["hop_index"] == 1


def test_write_and_reread_auth_csv_matches_parse_order(tmp_path):
    rows, _ = Wr.campaign_to_rows([(1000, "U1@D", "F", "T1")], campaign_id=0, row_id_start=0)
    p = tmp_path / "a.csv"
    Wr.write_auth_csv(rows, str(p))
    # re-read exactly like real auth.txt: headerless, AUTH_FIELDS order
    df = pd.read_csv(p, header=None, names=Wr.AUTH_FIELDS)
    assert list(df.iloc[0][Wr.AUTH_FIELDS]) == \
        [1000, "U1@D", "U1@D", "F", "T1", "NTLM", "Network", "LogOn", "Success"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_writer.py::test_campaign_to_rows_constant_attrs_and_labels -q`
Expected: FAIL — `has no attribute 'campaign_to_rows'`.

- [ ] **Step 3: Add emission functions to `src/writer.py`**

```python
def campaign_to_rows(placed_events, campaign_id, row_id_start):
    """Placed campaign -> (auth row dicts, label dicts). Constant attributes;
    hop_index = within-campaign emission order (fan-out has no real hops)."""
    rows, labels = [], []
    for i, (t, user, src, dst) in enumerate(placed_events):
        rid = row_id_start + i
        rows.append({
            "time": int(t), "src_user": user, "dst_user": user,
            "src_computer": src, "dst_computer": dst,
            "auth_type": AUTH_TYPE, "logon_type": LOGON_TYPE,
            "auth_orientation": AUTH_ORIENTATION, "success": SUCCESS,
        })
        labels.append({"row_id": rid, "is_malicious": 1,
                       "campaign_id": int(campaign_id), "hop_index": i})
    return rows, labels


def write_auth_csv(rows, path):
    """Headerless, exact AUTH_FIELDS order (LANL auth.txt has no header)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r[k] for k in AUTH_FIELDS])


def write_labels_csv(labels, path):
    """Headed labels file, row-aligned to the auth file by row_id."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        w.writeheader()
        w.writerows(labels)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_writer.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/writer.py tests/test_writer.py
git commit -m "feat: day-3 LANL row + row-aligned label emission"
```

---

### Task 5: Windowed benign extraction

**Files:**
- Create: `src/benign.py`
- Test: `tests/test_benign.py`

`sample_benign` pulls a random benign sample from `auth.txt.gz` restricted to a time window, excluding rows whose `(src_computer, dst_computer)` is a known red-team edge (cheap label-noise filter). It reuses the Day-1 DuckDB relation builder. Uses DuckDB `USING SAMPLE reservoir(n ROWS)` with a fixed seed for reproducibility. The test drives it against a tiny in-memory table via the same SQL shape.

- [ ] **Step 1: Write the failing test**

```python
import duckdb
import pandas as pd
import pytest
import src.benign as B


def test_sample_benign_windows_and_filters_redteam_edges():
    con = duckdb.connect()
    con.execute("""CREATE TABLE auth AS SELECT * FROM (VALUES
        (100,'U1@D','U1@D','A','B','NTLM','Network','LogOn','Success'),
        (200,'U2@D','U2@D','A','C','NTLM','Network','LogOn','Success'),
        (5000,'U3@D','U3@D','A','D','NTLM','Network','LogOn','Success'),
        (150,'U9@D','U9@D','E','F','NTLM','Network','LogOn','Success')
      ) AS t(time,src_user,dst_user,src_computer,dst_computer,
             auth_type,logon_type,auth_orientation,success)""")
    redteam_edges = {("E", "F")}                    # this benign-looking row is a known attack
    df = B.sample_benign_from_relation(
        con, "auth", t_lo=0, t_hi=1000, n=10, redteam_edges=redteam_edges, seed=1)
    got = set(zip(df.src_computer, df.dst_computer))
    assert ("A", "B") in got and ("A", "C") in got   # in-window benign kept
    assert ("A", "D") not in got                      # time 5000 out of window
    assert ("E", "F") not in got                      # red-team edge filtered
    assert list(df.columns) == list(B.AUTH_COLS)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_benign.py -q`
Expected: FAIL — `module 'src.benign' has no attribute 'sample_benign_from_relation'`.

- [ ] **Step 3: Write `src/benign.py`**

```python
"""Day-3 benign sampler: windowed random draw from auth.txt.gz, excluding
known red-team edges. Feeds Day-4 negatives (fit window -> train, holdout -> test).
"""
import pandas as pd

from src.parse import AUTH_COLS, _auth_rel, open_con


def sample_benign_from_relation(con, relation, t_lo, t_hi, n, redteam_edges, seed):
    """Random benign sample from a DuckDB relation (table name or read_csv expr)
    within [t_lo, t_hi], excluding rows on known red-team edges. AUTH_COLS order."""
    cols = ", ".join(AUTH_COLS)
    con.execute(f"SELECT setseed({(seed % 1000) / 1000.0})")
    df = con.execute(
        f"SELECT {cols} FROM {relation} "
        f"WHERE time >= {int(t_lo)} AND time <= {int(t_hi)} "
        f"USING SAMPLE reservoir({int(n)} ROWS)"
    ).df()
    if redteam_edges:
        mask = [(s, d) not in redteam_edges
                for s, d in zip(df["src_computer"], df["dst_computer"])]
        df = df[mask].reset_index(drop=True)
    return df[list(AUTH_COLS)]


def sample_benign(auth_path, t_lo, t_hi, n, redteam_edges, seed, temp_dir=None):
    """Same, driven straight off auth.txt.gz via the Day-1 read_csv relation."""
    con = open_con(temp_dir=temp_dir)
    return sample_benign_from_relation(
        con, _auth_rel(auth_path), t_lo, t_hi, n, redteam_edges, seed)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_benign.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/benign.py tests/test_benign.py
git commit -m "feat: day-3 windowed benign sampler with red-team-edge filter"
```

---

### Task 6: Mixed-corpus builder

**Files:**
- Create: `src/mixed.py`
- Test: `tests/test_mixed.py`

`build_mixed_corpus` down-samples benign to hit a target malicious:benign `rate`, concatenates the malicious rows, sorts by `time` (realistic interleave), and returns the mixed auth DataFrame plus a row-aligned labels DataFrame (`is_malicious` 1 for synthetic, 0 for benign). Benign carries no campaign — labelled `campaign_id=-1, hop_index=-1`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import pytest
import src.mixed as M
import src.writer as Wr


def _mal_rows(n):
    rows, _ = Wr.campaign_to_rows(
        [(t, "U1@D", "F", f"T{t}") for t in range(n)], campaign_id=0, row_id_start=0)
    return pd.DataFrame(rows)


def _benign_rows(n):
    return pd.DataFrame([{
        "time": 1000 + i, "src_user": "Ub@D", "dst_user": "Ub@D",
        "src_computer": "X", "dst_computer": "Y", "auth_type": "Kerberos",
        "logon_type": "Network", "auth_orientation": "LogOn", "success": "Success"}
        for i in range(n)])


def test_build_mixed_corpus_hits_rate_and_aligns_labels():
    mal = _mal_rows(10)
    benign = _benign_rows(100000)
    mixed, labels = M.build_mixed_corpus(mal, benign, rate=0.001, seed=3)
    # rate = mal/benign -> ~10/0.001 = ~10000 benign kept
    n_benign = int((labels["is_malicious"] == 0).sum())
    n_mal = int((labels["is_malicious"] == 1).sum())
    assert n_mal == 10
    assert abs(n_benign - 10000) < 500                 # ~ target rate
    assert len(mixed) == len(labels)                    # row-aligned
    assert list(mixed["time"]) == sorted(mixed["time"]) # sorted interleave
    assert set(labels["is_malicious"].unique()) == {0, 1}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mixed.py -q`
Expected: FAIL — `module 'src.mixed' has no attribute 'build_mixed_corpus'`.

- [ ] **Step 3: Write `src/mixed.py`**

```python
"""Day-3 mixed corpus: inject synthetic malicious rows into real benign at a
target base rate, sorted by time. Labels row-aligned (1 malicious, 0 benign)."""
import numpy as np
import pandas as pd

import src.writer as Wr

_LABEL_COLS = ["row_id", "is_malicious", "campaign_id", "hop_index"]


def build_mixed_corpus(mal_df, benign_df, rate, seed):
    """rate = malicious:benign. Down-sample benign to n_benign = n_mal / rate,
    concat, sort by time. Returns (mixed_auth_df, labels_df), row-aligned."""
    rng = np.random.default_rng(seed)
    n_mal = len(mal_df)
    n_benign = min(len(benign_df), int(round(n_mal / rate)))
    idx = rng.choice(len(benign_df), size=n_benign, replace=False)
    benign = benign_df.iloc[idx].copy()

    mal = mal_df[Wr.AUTH_FIELDS].copy()
    mal["is_malicious"] = 1
    benign = benign[Wr.AUTH_FIELDS].copy()
    benign["is_malicious"] = 0

    mixed = pd.concat([mal, benign], ignore_index=True)
    mixed = mixed.sort_values("time", kind="stable").reset_index(drop=True)

    labels = pd.DataFrame({
        "row_id": np.arange(len(mixed)),
        "is_malicious": mixed["is_malicious"].to_numpy(),
        "campaign_id": -1, "hop_index": -1,
    })[_LABEL_COLS]
    mixed = mixed[Wr.AUTH_FIELDS].reset_index(drop=True)
    return mixed, labels
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_mixed.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/mixed.py tests/test_mixed.py
git commit -m "feat: day-3 mixed-corpus builder with base-rate down-sampling"
```

---

### Task 7: Orchestrator — produce all Day-3 artifacts + exit criteria

**Files:**
- Create: `scripts/run_day3.py`

Loads Day-1/2 artifacts, regenerates campaigns with the creds fix, places + writes `synth_auth.csv` + `synth_labels.csv`, extracts both benign windows off `auth.txt.gz`, builds a mixed corpus per sweep rate, then asserts the SPEC §3 DAY 3 exit criteria.

- [ ] **Step 1: Write `scripts/run_day3.py`**

```python
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

    graph = P.load_graph(p["graph"])
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
    for br in d3["base_rates"]:
        mixed, labels = M.build_mixed_corpus(mal_df, ben_fit, rate=br["rate"], seed=cfg["seed"])
        Wr.write_auth_csv(mixed.to_dict("records"),
                          d3["synth_auth"].replace("synth_auth.csv", f"mixed_{br['name']}.csv"))
        labels.to_csv(d3["synth_labels"].replace("synth_labels.csv",
                                                 f"mixed_{br['name']}_labels.csv"), index=False)
        print(f"mixed {br['name']}: {len(mixed):,} rows  mal={int((labels.is_malicious==1).sum()):,}")

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
    print(f"[x] mixed corpora built at {[b['name'] for b in d3['base_rates']]}")
    print(f"[i] business-hours (6-17) fraction of synthetic events = {biz_frac:.2f} (real red team ~1.0)")
    print(f"[i] discard_rate={stats['discard_rate']:.3f} cap_rate={stats['cap_rate']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the orchestrator**

Run: `python -m scripts.run_day3`
Expected: benign-extraction timing (one pass over the 7.1 GB gz — minutes), mixed-corpus lines, and the exit-criteria checklist with the four `[x]` boxes checked and `violations=0`. The business-hours fraction should be high (start times are drawn ∝ hourly volume, which peaks 7–15). If `benign_n` exceeds available in-window rows, the sampler returns fewer — that's fine; note it.

- [ ] **Step 3: Confirm the whole suite still passes**

Run: `python -m pytest -q`
Expected: PASS (parse + profile + walker + writer + benign + mixed).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_day3.py
git commit -m "feat: day-3 orchestrator — synth rows, benign windows, mixed corpora, exit checks"
```

---

## Day-3 Exit Criteria (SPEC §3 DAY 3) — verify before the phase-gate

- [ ] `synth_auth.csv` parses cleanly with the same reader used for real `auth.txt` (`P.AUTH_COLS`, headerless).
- [ ] Field-level value domains are a subset of those seen in real data (constant categoricals; real computer/user tokens by construction).
- [ ] `synth_labels.csv` aligns row-for-row with the data file.
- [ ] Mixed corpus generated at each configured base rate (1e2/1e3/1e4).
- [ ] (V1 preview) every emitted `(src,dst)` is a real graph edge — 0 violations.

**Do not start Day 4.** SPEC §3.0 mandates a grilling checkpoint (review Day-3 output, pin the Day-4 augmentation/eval knobs — scarcity-curve campaign counts, detector family, feature set + ablation, memorization metric, seed count for CIs — explicit go/no-go) before the evaluation layer.

---

## Self-Review

- **Spec coverage (SPEC §3 DAY 3):** 3.1 attributes (constant tuple) → Task 4 (`campaign_to_rows` constants). 3.2 time placement (hourly-weighted start + Δt→absolute) → Task 3. 3.3 writer (exact LANL order, labels, mixed corpus) → Tasks 4 (rows/labels) + 6 (mixed) + 7 (orchestrate). Exit criteria (parses, domain subset, label alignment, mixed at base rate) → Task 7. Benign for downstream eval (not in the original SPEC §3 but required by the augmentation decision) → Task 5. Gap #2 fix (grilling decision) → Task 2.
- **Placeholder scan:** every code step is complete runnable code; the two `[i]` lines in Task 7 are informational diagnostics, not unchecked deliverables.
- **Type consistency:** `AUTH_FIELDS` (writer) == `AUTH_COLS` (parse) field order, cross-checked in Task 4's re-read test. `campaign_to_rows(placed, campaign_id, row_id_start) -> (rows, labels)` used identically in Task 4 tests and the Task 7 loop. `sample_benign(auth_path, t_lo, t_hi, n, redteam_edges, seed)` — the `day3.fit_window`/`holdout_window` are 2-lists spread as `t_lo, t_hi` in Task 7. `build_mixed_corpus(mal_df, benign_df, rate, seed) -> (mixed, labels)` — same in Task 6 test and Task 7. Synth event tuple `(dt_offset, user, src, dst)` from Day-2 consumed unchanged by `place_campaign`. `generate_campaign` signature unchanged by Task 2 (uses existing `dists["creds_per_campaign"]`), so all Day-2 tests and the Task 7 call site still match.
- **Known ceilings (`ponytail:`):** benign label noise filtered only on exact `(src,dst)` red-team edges — undetected attacks on other edges remain (documented, Day-4 limitations). Multi-origin (#5) and cred→target proxy (#1) deferred by decision. Start-time day is uniform across the 58-day window while real red-team activity concentrated in the first ~30 days — acceptable for benign-blended augmentation; tighten only if Day-4 shows a temporal artifact.
