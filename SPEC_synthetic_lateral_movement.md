# SPEC: Synthetic Malicious Authentication Log Generator (LANL Lateral Movement)

**Version:** 1.2 (revised after Day-1 phase-gate — empirical data showed the labeled red-team activity is **credential-reuse fan-out from a foothold**, not multi-hop path traversal; attributes are constant, not skewed. Structure generator, attribute layer, and framing updated. See `docs/superpowers/phase-gate-day1-to-day2.md` for the measured evidence.)

> **v1.2 empirical correction (SETTLED at Day-1 gate — do not relitigate):** Measured on the recovered 701 red-team events: `chained_links = 0` in all 19 campaigns (no hop where a target becomes the next source). Dominant shape is **one source host → many high-in-degree targets** (top-2 campaigns are 1→168 and 1→88 fan-outs, 72% of fit mass). Attributes are **100% NTLM / Network / LogOn**, 700/701 Success — degenerate, nothing to sample. Consequences threaded through §1, §3 DAY 2, §3.1, §7 below.
**Timeline:** 4 working days, hard stop
**Audience:** downstream implementation agent
**Status of decisions below:** SETTLED. Do not relitigate the architecture. If you believe a decision is wrong, flag it and continue with the specified approach unless told otherwise.
**Execution is phase-gated:** do not roll from one day into the next without the grilling checkpoint in §3.0.

---

## 0. Objective

Produce a generator that emits **large volumes of synthetic malicious authentication events** simulating lateral movement across the LANL enterprise network, in the exact schema of the LANL `auth.txt` file, such that:

1. Every generated host-to-host authentication corresponds to a connection that **actually exists** in the real network graph (zero hallucinated edges).
2. The statistical character of the attack chains (length, dwell time, auth protocol mix, target selection) resembles the real red-team activity.
3. A detector trained on the synthetic data transfers to **real, held-out** red-team events.

Criterion 3 is the deliverable that matters. Criteria 1 and 2 are means to it.

### 0.1 Scale target and environment

- **Volume target:** ≥ 100,000 synthetic malicious events across ≥ 5,000 distinct campaigns — roughly two orders of magnitude beyond the 749 real events. Generation should take minutes, not hours; if it does not, the walk implementation is wrong.
- **Environment:** Python 3.11+ on **Windows** (verified: 3.11.9, `networkx` 3.4.2 / `numpy` 2.0.0 / `pandas` 2.2.2 / `scipy` 1.14.1 / `scikit-learn` 1.5.1 / `matplotlib` 3.9.2 / `pyyaml` present). Core dependencies: `networkx`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, **`duckdb`** (the streaming engine — see §3 Day-1 task 2; `zcat`/`awk` are unavailable on Windows). `sdv` only if §6 is attempted. No GPU required at any point on the critical path.
- **Disk:** budget ~15 GB for the compressed source files. Never materialise the uncompressed `auth.txt`.

---

## 1. Architecture Decision Record

### 1.1 The decision

**Decompose the problem into structure and attributes, and solve each with the cheapest sufficient method.**

| Layer | What it produces | Method |
|---|---|---|
| **Structure** | The attack **fan-out**: from a foothold, which real targets are authenticated into, with which credentials | Credential-aware weighted **fan-out** over a `NetworkX` graph built from real logs — seed a foothold, sample a fan-out breadth, branch to real out-edge targets weighted by in-degree (NOT a linear path walk; the labeled data has zero multi-hop pivoting — see v1.2 note) |
| **Attributes** | Per-event fields: auth type, logon type, orientation, success/failure, inter-hop Δt | Empirical conditional sampling from a lookup table fitted on real red-team events |
| **Placement** | Where campaigns land in wall-clock time, interleaved with benign traffic | Sampling weighted by the benign activity profile |

No neural network is on the critical path. TVAE is a **stretch goal only** (§6).

### 1.2 The reasoning

**The 749 red-team events are not 749 independent samples.** They are a small number of campaigns produced by one operator with one toolkit. The effective sample size, measured in distinct *behaviors*, is single digits. This single fact eliminates every deep generative option: there is not enough signal to fit one, and any model that appears to fit is memorizing.

**Only half the problem is hard.** Structure is where hallucination risk lives and where a reviewer's eye goes. Attributes are ~5 low-cardinality categoricals plus one numeric; conditioned on the edge, empirical sampling is *more* faithful than a model trained on 749 rows, and takes an afternoon.

### 1.3 Rejected alternatives — record these so they are not re-proposed

| Option | Why rejected |
|---|---|
| **TabDDPM** (tabular diffusion) | Genuinely strong on tabular benchmarks, but generates rows **i.i.d.** — no notion of sequence or graph. It would happily emit `C1→C2` followed by `C9→C4` with no connecting path, so the random walk would still be required underneath it. Pays diffusion training cost and research-grade config wrangling to solve the half of the problem that was already easy. Its benchmark margin over TVAE is within noise at n≈749, and it will overfit. |
| **GReaT** (LLM-based tabular generation) | Its entire premise is that a pretrained LM already understands column semantics (`age`, `income`, `city`) and transfers that prior. **LANL is anonymized** — `C17693$@DOM1`, `U8946`, `ANONYMOUS LOGON`. There is no semantic prior to exploit. Meanwhile ~17k hostnames tokenize into meaningless subword fragments, autoregressive sampling is slow at the volumes required, and nothing constrains output to a valid edge. All of the cost, none of the benefit. |
| **CTGAN / standard GANs** | Mode collapse risk at this sample size. |
| **GCN / GraphSAGE** | GCN is transductive and memory-heavy at 17k nodes; GraphSAGE is inductive and viable in principle but is multi-week engineering, not 4-day. |
| **Agentic LLM workflow** | Token cost, latency, and no mechanism to hold a 17k-node topology in context reliably. |

---

## 2. Data

### 2.1 Source

Los Alamos National Laboratory, *Comprehensive, Multi-Source Cyber-Security Events* (Kent, 2015).
Portal: `https://csr.lanl.gov/data/cyber1/`

Required files:

- `auth.txt.gz` — authentication events. **This is the large one.** Roughly 1.05 billion events; on the order of ~11–12 GB gzipped and ~70 GB uncompressed. *Verify actual sizes on download rather than trusting these figures.*
- `redteam.txt.gz` — the 749 labelled malicious events.

Optional, not required for this POC: `proc.txt.gz`, `flows.txt.gz`, `dns.txt.gz`.

### 2.2 Schemas

`auth.txt` — comma-separated, no header:

```
time,src_user@src_domain,dst_user@dst_domain,src_computer,dst_computer,auth_type,logon_type,auth_orientation,success_or_failure
```

`redteam.txt` — comma-separated, no header:

```
time,user@domain,src_computer,dst_computer
```

Notes the implementer must internalise:

- `time` is **integer seconds from the start of the collection period**, beginning at 1. It is not a Unix epoch. The collection spans 58 days.
- Missing/redacted values appear as `?`. Handle them as a real category, do not drop them.
- Roughly 12,425 users and 17,684 computers appear. *Verify these counts during the parse rather than hard-coding them.*
- `redteam.txt` rows are a **subset** of `auth.txt` rows — join on `(time, user, src_computer, dst_computer)` to recover the full 9-field record for each malicious event.

### 2.3 The train/test split — do this on Day 1, before anything else

Split the red-team events **by campaign / time, never by row**, and quarantine the test portion immediately.

- Fit split: earliest ~70% of the red-team **timeline** (30% is measured on the **time axis**, not by event count).
- Held-out split: latest ~30% of the timeline.

**Split mechanic (SETTLED):** **whole-campaign assignment** — order the campaigns (§3 Day-1 t6, `campaign_gap = 4h`) by start time and cut at the inter-campaign **gap nearest the 70%-of-timeline mark**. Assign each *entire* campaign to one side; never slice a campaign across the boundary ("straddle"). Because campaigns are gap-separated intervals, the cut lands in a gap and no campaign straddles — verified on the real data, and this stays straddle-proof even if `campaign_gap` is later flipped to 1h (a raw-timestamp cut is not).

**Measured reality on `redteam.txt` (v1.2 — actuals from the executed Day-1 run):** 749 raw rows → 715 after dedup → **701 recovered** (14 unmatched, all from foothold C17693). **19 campaigns** (13 fit / 6 holdout); the 70%-timeline cut yields **fit = 650 events / holdout = 51 events (7.3% by event mass)**. Top-2 fit campaigns = **261 and 207 events ≈ 72% of fit mass**, both early fan-outs. (The v1.1 estimates of ~690/59 and 273/209 predated dedup — superseded by these.) The imbalance is intrinsic: campaign sizes are extremely skewed (top two campaigns = 273 and 209 events ≈ 64% of all 749, both early). Do NOT rebalance by switching to a 30%-of-*events* split — that would drag an early mega-campaign into the holdout, break the "holdout = latest" property, and starve the fit set of its richest signal. Consequence: TSTR runs on ~59 real positives and will be **noisy — report AUC-PR with that caveat plainly**.

Every distribution, weight, and lookup table in the generator must be fitted on the **fit split only**. The held-out split is touched exactly once, on Day 4, for TSTR evaluation.

> This is the single most common way a project like this produces a meaningless result. Row-level shuffling leaks campaign structure across the split and inflates every downstream number. Write the split to disk as a frozen artifact on Day 1 and do not regenerate it.

---

## 3. Day-by-day execution plan

Each day has **exit criteria**. Do not begin the next day until they pass. If a day slips, cut scope from Day 3, never from Day 4.

### 3.0 Phase-gate grilling protocol (MANDATORY between days)

**Do NOT auto-roll from one day into the next.** Before starting any day, STOP and run a grilling checkpoint with the operator:

1. **Review prior-day state** — walk the exit criteria and artifacts produced; surface anything that drifted, failed, or looks off. The operator may revise the previous day before it is considered done.
2. **Revise the next-day plan** — re-open the upcoming day's assumptions and knobs in light of what the last day actually produced; grill the operator and adjust.
3. **Explicit go/no-go** — the operator decides whether to proceed and deepens their understanding of the pipeline. No day begins without this confirmation.

Applies before Day 1→2, 2→3, 3→4, and before the §6 TVAE stretch.

---

### DAY 1 — Acquisition, parse, graph, profile

**Deliverables:** `data/` populated, `graph.pkl`, `redteam_profile.json`, frozen split files.

**Tasks**

1. **Download** `auth.txt.gz` and `redteam.txt.gz`. Record checksums. Do not decompress to disk — stream from gzip throughout.

2. **Single streaming pass over `auth.txt`.** One pass, accumulating everything needed. Do not plan on multiple passes; each one costs hours.

   Performance guidance: naive Python line-parsing of ~1B rows is slow (hours). **Engine decision: `duckdb`** — it reads `auth.txt.gz` directly, streams, and runs the group-bys (edge/user-host counts, degrees, hourly volume, marginals) plus the redteam↔auth join in C++ SQL. (`zcat | awk` is not available on Windows; `polars` is a viable alternative but DuckDB does the join in the same pass.) Benchmark on the first ~10M rows and extrapolate before committing to a full pass.

   Accumulate in that one pass:
   - `edge_counts[(src_computer, dst_computer)] → int`
   - `user_host_counts[(user, computer)] → int` — which credentials are seen on which hosts
   - `node_in_degree`, `node_out_degree`
   - `hourly_volume[hour_of_day] → int` — the benign activity profile, for campaign placement
   - Global marginal distributions of `auth_type`, `logon_type`, `auth_orientation`, `success/failure`

   Memory is not a concern: ~17.7k nodes and a few million unique edges fit comfortably in RAM. Encode host and user strings to integer IDs to keep it compact.

3. **Build the graph.** `networkx.DiGraph`. Nodes = computers, carrying `in_degree`, `out_degree`, `in_out_ratio`, and a derived `is_server` binary. **Target-attractiveness (SETTLED — revised):** do NOT hard-commit to a "server" label (the anonymized data has no role field; "server" is an *inference*, not a fact). Fit the walk's target-attractiveness to the **empirical in-degree (+ in:out ratio) distribution of the fit-split red-team targets**, using structural features only — never host identities, or it overfits. `is_server` is a *derived* binary (default: top-decile in-degree combined with high in:out ratio — a server receives far more than it initiates) kept only for the Day-3 attribute key; backoff covers it if weak. **Day-1 validation (in-flight, needs the parsed graph):** compare the in-degree / in:out distribution of the **301 real target hosts** against all hosts; if targets are clearly high in-degree, keep the proxy and set the threshold where it separates them; if the signal is muddy, drop the "server" proxy and weight candidates directly by similarity to the empirical target-profile. Edges carry `weight = observed count`.

   **Optional — hub observability (increases model observability; NOT an exit criterion):** emit a short report of the top-N hubs (by in-degree, in:out ratio, and — if cheap — centrality) so the operator can *see* which nodes dominate the graph, and on Day 2 which hubs the walks funnel through. Makes the "popular hub" structure that drives target selection legible.

4. **Recover full records for red-team events** by joining `redteam.txt` against `auth.txt` during the same pass.

5. **Freeze the split** (§2.3). Write `redteam_fit.csv` and `redteam_holdout.csv`.

6. **Profile the fit split** into `redteam_profile.json`:
   - **Fan-out breadth distribution (distinct targets per campaign) — the Day-2 driver (v1.2).** The old "chain length = hops per campaign" is retained only as a descriptive stat; because `chained_links = 0`, "chain length" is really fan-out breadth, and Day-2 samples *breadth*, not path depth. Also record **credentials-per-campaign** (distinct creds vs. targets) — the reuse pattern Day-2 reproduces.
   - Chain length distribution (hops per campaign). **Campaign definition (REVISED):** sessionize the red-team event stream **globally by inter-event time gap** — a contiguous burst of activity is one campaign. **`campaign_gap = 4h`**, chosen as the empirical valley in the real gap distribution (measured from `redteam.txt`: median inter-event gap 1.7 min, 90% under 18 min, then a jump to hours; 4h–8h thresholds all yield ~18–19 campaigns — a plateau that marks a real session boundary), median chain length ~9. **1h fallback** if synthetic chains look too long — a cheap config flip. *Rejected the earlier "shared credential" grouping: real intrusions harvest and swap credentials, which shatters one campaign into fragments.* **This is a low-stakes definitional knob:** the walker samples chain length from this distribution (Day-2 step 5), so V2 chain-length matches by construction and will not auto-flag a bad gap; the TSTR detector is per-event, so it is largely gap-insensitive. Do NOT group per-credential or per-host (only ~4 source hosts, one dominant).
   - Inter-hop Δt distribution (keep the empirical values, do not fit a parametric form yet)
   - Seed-host characteristics (out-degree, in-degree of where chains start)
   - Target-host characteristics (are targets high in-degree? by how much?)
   - Conditional attribute distributions (§4.2)
   - Credential reuse pattern: how many distinct credentials per campaign, when do new ones appear

**Exit criteria**

- [ ] Graph loads from disk; node and edge counts printed and sanity-checked against §2.2 figures
- [ ] All red-team events successfully joined to their full 9-field records; any that fail to join are counted and explained
- [ ] Split files written and immutable
- [ ] `redteam_profile.json` contains every distribution listed above

---

### DAY 2 — The fan-out generator

**Deliverable:** `walker.py` (name kept for continuity) producing valid attack **campaigns** as sets of `(time_offset, user, src_host, dst_host)` tuples — a foothold fanning out to real targets.

**v1.2 reframe (SETTLED):** the labeled red-team activity is **not** a multi-hop path — `chained_links = 0` across all 19 campaigns. It is **credential-reuse fan-out**: one foothold authenticates outward to many high-in-degree targets. The generator branches from a foothold; it does NOT walk a path. Optional shallow depth (a target that itself becomes a second-tier foothold) only if the mixed campaigns warrant — default off.

**The campaign state** is `(foothold_host, compromised_credentials: set, targets_hit: list)`. The realism driver is the *breadth* of the fan-out and *which* credentials reach *which* targets — not path length.

**Step algorithm**

1. **Seed the foothold.** Pick a foothold host by sampling from the empirical seed-host profile (real footholds are high-out-degree operator hosts, e.g. the C17693-like source). Assign an initial credential set seen on that host from `user_host_counts`.

2. **Harvest.** Load the credentials observed on the foothold (from `user_host_counts`) into the compromised set, with a cap. Real footholds carry many credentials (camp 5: 45 distinct) — a fan-out that reuses one credential looks nothing like the real data.

3. **Fan-out breadth.** Sample the number of targets `k` from the empirical **per-campaign dst-count (breadth)** distribution (real values: 168, 88, 46, 17, 12, …). This replaces the old "chain length" — breadth, not depth.

4. **Target selection & weighting.** Candidate targets are the real out-edges from the foothold. **Hard constraint: no target may be an edge absent from the graph.** Draw `k` targets (without replacement) weighted by:

   ```
   score = edge_weight^alpha  ×  target_attractiveness^beta  ×  credential_bonus
   ```

   - `alpha` biases toward well-trodden edges (blending in)
   - `beta` biases toward high in-degree servers and domain controllers (the objective)
   - `credential_bonus` multiplies candidates where a compromised credential has actually been observed on that edge

   **On the credential constraint:** an exact observed `(user, src, dst)` triple is too sparse. Use it as a **weight bonus, not a filter**. The hard filter stays at the host-edge level. Document this in the README — a reviewer will ask. Assign the per-target credential by sampling among compromised creds observed on that target's edge (backoff: any compromised cred).

5. **Terminate** when `k` targets are emitted, or when the foothold's real out-edges are exhausted (fan-out is capped by real out-degree — a dead-end here means the foothold simply has fewer than `k` real targets; take all of them, do not fabricate edges).

6. **Assign Δt** per target by bootstrapping from the empirical inter-event Δt distribution; timestamps are offsets within the campaign window.

**Guard rails**

- If a sampled `k` exceeds the foothold's real out-degree, cap `k` at the out-degree (or reseed to a higher-out-degree foothold). Log how often this fires — frequent capping means the seed-host profile is drawing footholds too small.
- No path-oscillation concern (no path). If optional shallow depth is enabled, prevent revisiting the foothold.

**Tuning:** adjust `alpha` and `beta` until generated fan-out-breadth and target-in-degree distributions overlap the fit split. A handful of manual iterations, not an optimization job. Timebox to two hours.

**Exit criteria**

- [ ] 10,000 campaigns generate in under a minute
- [ ] Assertion passes: every generated `(foothold, target)` edge exists in the graph
- [ ] Generated fan-out-breadth distribution visually overlaps the real fit-split breadth distribution
- [ ] Credential-reuse per campaign (distinct creds vs. targets) overlaps the fit split
- [ ] Foothold-capping rate logged and under control

---

### DAY 3 — Attribute layer, time placement, output writer

**Deliverable:** `synth_auth.csv` in exact LANL format + `synth_labels.csv`.

**3.1 Attribute sampling**

**v1.2 correction (SETTLED):** the fit-split attributes are **not skewed — they are constant.** Measured: **100% NTLM, 100% Network, 100% LogOn**, 700/701 Success. The conditional lookup table is therefore **degenerate**: every cell resolves to `(NTLM, Network, LogOn, Success)`. Do **not** build the conditional machinery below — emit the constant tuple. Keep a one-line guard: if a future fit split shows >1 value in any attribute, fall back to the conditional table (kept for reference):

```
(dst_is_server, hop_index_bucket, auth_orientation)  →  distribution over (auth_type, logon_type, success/failure)
   backoff:  full key → drop hop_index → drop dst_is_server → global red-team marginal
```

The near-constant Success (700/701) and 100% NTLM/Network are the **fingerprint of remote credential replay** (network logon = remote auth, NTLM = the hash protocol), not noise. This constancy is **tool-specific** to this red team — a detector must not over-rely on it (see §7 Future Work). If your fitted table shows variety instead of constants, you have a bug in the join.

**3.2 Time placement**

Sample campaign start times weighted by `hourly_volume` so campaigns land inside plausible activity windows, and confirm the resulting temporal footprint resembles the real red team's. Convert per-hop Δt into absolute `time` values in LANL's seconds-from-start convention.

**3.3 Writer**

Emit rows in exact `auth.txt` field order (§2.2). Separately emit `synth_labels.csv` with `row_id, is_malicious, campaign_id, hop_index` so evaluation has ground truth without polluting the data file.

Also produce a **mixed corpus**: synthetic malicious events interleaved into a sample of real benign traffic at a configurable base rate. Default the rate to match the real malicious-to-benign ratio, which is extremely low. Make it a parameter — evaluation will want to sweep it.

**Exit criteria**

- [ ] Output parses cleanly with the same reader used for real `auth.txt`
- [ ] Field-level value domains are a subset of those seen in real data
- [ ] Labels file aligns row-for-row with the data file
- [ ] Mixed corpus generated at the default base rate

---

### DAY 4 — Validation. Do not compress this day.

**Deliverable:** `validation_report.md` with figures. This is what gets shown.

**V1 — Structural integrity (hard assertions, must be 100%)**

- Every generated `(src, dst)` exists in the real graph
- Timestamps within a campaign are strictly increasing; all Δt > 0
- No credential is used at a hop before the hop where it was acquired
- No campaign extends beyond the 58-day collection window

Any failure here is a bug, not a tuning issue. Fix before proceeding.

**V2 — Distributional fidelity (compare synthetic vs. the fit split)**

- **Fan-out breadth** (targets per campaign), inter-event Δt, credentials-per-campaign: two-sample KS test
- Target in-degree: Jensen–Shannon divergence, target < 0.1
- Categorical fields (auth_type, logon_type, orientation, success): both sides are constant `(NTLM, Network, LogOn, Success)` → JS divergence trivially ~0. Report it, but note it proves nothing (a constant matching a constant) — the real fidelity test is structural (breadth, target in-degree, credential reuse).
- Plot each as overlaid histograms

**V3 — TSTR (Train on Synthetic, Test on Real) — the headline number**

1. Train a detector on **synthetic malicious + real benign**. Keep the model simple and boring — gradient-boosted trees or logistic regression over per-event features (edge rarity, dst in-degree, auth type, logon type, hour of day, credential novelty on host). A weak model makes the data quality legible; a strong model hides it.
2. Test on the **held-out real red-team split + real benign**, untouched since Day 1.
3. Report precision, recall, AUC, and precision-recall curve. Given the extreme class imbalance, **AUC-PR is the number to lead with; ROC-AUC will look flatteringly high and mean little.**

**Required baselines for context** — a TSTR number alone is uninterpretable:

| Baseline | Purpose |
|---|---|
| Train on real fit-split red team, test on held-out | The ceiling you are trying to approach |
| Train on naive uniform random walks (no weighting, no credential model) | Proves the weighting and credential layers earn their place |
| Simple unsupervised rarity heuristic, no training | Proves the whole pipeline beats doing nothing |

**Exit criteria**

- [ ] All V1 assertions pass at 100%
- [ ] V2 figures produced for every listed field
- [ ] V3 table complete with all three baselines
- [ ] Report states plainly where the synthetic data falls short — an honest limitations section is worth more than a flattering number

---

## 4. Repository layout

```
├── data/                        # gitignored; raw + derived
├── src/
│   ├── parse.py                 # Day 1: streaming pass, graph + profile build
│   ├── profile.py               # Day 1: red-team characterisation, split freeze
│   ├── walker.py                # Day 2: credential-aware weighted random walk
│   ├── attributes.py            # Day 3: conditional sampling with backoff
│   ├── writer.py                # Day 3: LANL-format emission, benign interleave
│   └── validate.py              # Day 4: V1/V2/V3
├── config.yaml                  # alpha, beta, chain length caps, base rate, campaign_gap (default 4h), seeds
├── notebooks/                   # exploration only, nothing load-bearing
└── validation_report.md
```

Set and record a random seed for every stochastic component. The results must be reproducible.

---

## 5. Risk register and fallbacks

| Risk | Trigger | Fallback |
|---|---|---|
| Full `auth.txt` pass is too slow | Day 1 benchmark extrapolates beyond ~4 hours | Build the graph from a contiguous time-slice (e.g. first 20 days) covering the fit-split red-team window. Document the reduced coverage. |
| Download unavailable or throttled | Day 1 morning | Escalate immediately — this blocks everything. Do not spend more than 2 hours fighting it before raising it. |
| Walks dead-end frequently | Discard rate > 20% | Lower `beta`; if that fails, allow restart-from-seed mid-chain rather than relaxing the edge constraint. |
| Attribute cells too sparse | Most samples hitting the last backoff level | Accept it and say so in the report. Do not invent data to fill cells. |
| TSTR result is poor | Day 4 | **Report it.** A negative result with clean methodology is a legitimate POC outcome and more useful than a tuned-until-pretty number. Diagnose whether the failure is structural or attributional and say which. |

---

## 6. Stretch goal — TVAE attribute layer (only if Day 3 finishes early)

Swap `attributes.py` for a `TVAE` from the `SDV` library, fitted on the fit-split red-team records conditioned on hop context. The interface is identical, so it is a drop-in.

**Conditions:** attempt only if Days 1–3 exit criteria are all met with time to spare, and only as an *addition* — the empirical sampler remains the shipped path. If attempted, report both side by side under the same V2/V3 protocol. Expect little or no improvement; the value is in being able to say the comparison was run.

**Do not** let this consume Day 4. Validation is not negotiable; TVAE is.

---

## 7. Definition of done

1. `synth_auth.csv` generated at scale in exact LANL schema, with aligned labels.
2. 100% pass on all V1 structural assertions.
3. `validation_report.md` containing V2 figures, the V3 TSTR table with all three baselines, and an honest limitations section.
4. `README.md` — **audience: technical reviewer/grader; ~2 pages + an ASCII pipeline diagram; concise; do NOT include the 4-day timeline.** Fixed 6-section structure:
   1. **Overall Goal** — a short POC completable within a week (scoping → brainstorming → execution → result).
   2. **Scoping** — the authentication slice of IAM, specifically user login/logout sessions; synthesizing malicious logging behavior that maps real-world lateral movement. **Frame it accurately (v1.2):** the labeled red-team activity is **credential-reuse fan-out from a foothold** (one source → many high-in-degree targets), **not** multi-hop path traversal — the data shows zero pivoting. Fan-out is a form of lateral movement; do not claim a multi-hop chain the data lacks. State plainly that mechanism (credential theft / pass-the-hash) is *interpretation* — LANL is anonymized auth events showing structure and outcome, not mechanism or intent; this is a sanctioned red-team exercise (the foothold is likely an operator console, credentials likely provided).
   3. **Dataset & Model Choice** — brief LANL intro; the two current choices (NetworkX weighted-walk + empirical sampler, and the TVAE extension).
   4. **Architecture** — engineering flow from dataset input → processing → synthesis (pipeline diagram + config table: alpha, beta, base_rate, campaign_gap, seeds). **Must also state the two required decisions:** (a) the fit/holdout split methodology, and (b) enforcing credential realism as a *weight*, not a hard filter.
   5. **Evaluation** — how synthetic quality is judged; the TSTR protocol (lead with AUC-PR). **State the transfer ceiling honestly (v1.2):** fit and holdout are the *same* red team (same toolkit, same 100%-NTLM fingerprint, same fan-out shape), so TSTR demonstrates transfer *within one operator's tradecraft*, not to novel attackers. Lead with the **rarity-heuristic baseline comparison** — because the per-event signal reduces to (rare edge + high-in-degree target + credential novelty), the synthetic generator's uplift over a dumb rarity scorer may be modest; report that plainly (SPEC §5 accepts a modest/negative result with clean methodology).
   6. **Future Work** — **generalization beyond one operator (v1.2, lead item):** the attribute constancy (100% NTLM/Network) is this red team's toolkit fingerprint, not a universal malicious signature; demonstrating transfer to a *different* attacker (Kerberos pass-the-ticket, WMI, low-and-slow single-target) is the key untested direction. **Optional shallow-depth fan-out:** promote a first-tier target to a second-tier foothold if real multi-tier structure is later found. Then: why CTGAN / GCN / GraphSAGE / TabDDPM / GReaT were not chosen and the conditions under which each could win; **richer server/target detection via graph centrality (betweenness, PageRank) — deferred as overkill for a 4-day POC, but a stronger signal than degree alone**; **optional hub-observability instrumentation** (surface the most "popular" hubs from the graph and how the walks traverse them, to make target selection legible during modelling); simple statistical-correlation checks between data points; and privacy/memorization guardrails: Distance-to-Closest-Record (DCR / data copying), Membership Inference, and Attribute Inference.
   Throughout, make plain that **no deep generative model is on the critical path, and why** (see §1.2).
