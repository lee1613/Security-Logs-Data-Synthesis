# Phase-Gate: Day 1 → Day 2 (grilling checkpoint output)

**Date:** 2026-08-08. Status: Day-1 exit criteria PASS. Day 2 not started (SPEC §3.0 gate).

## Day-1 verified state (measured on real artifacts)

| Item | SPEC expected | Actual | Note |
|---|---|---|---|
| Graph | ~17,684 computers | 17,666 nodes / 419,744 edges | full pass confirmed complete |
| auth.txt.gz size | ~11–12 GB | 7.1 GB | SPEC estimate high; node count proves completeness |
| Red-team recovery | 749 subset | 749 raw → 715 dedup → **701** matched (34 dup, 14 unmatched) | 14 unmatched all from C17693 |
| Split | ~19 campaigns, ~8% holdout | 19 campaigns (13 fit / 6 holdout), 650 fit / 51 holdout, 7.3% | consistent |
| Target validation | targets high in-degree | median 13 vs 4, ratio 3.25, keep_server_proxy=true | proxy kept |
| Attributes | "skew to NTLM/Network" | **100% NTLM / Network / LogOn**, 700/701 Success | degenerate, not skew |

## Findings that reshape Day 2/3 (three SPEC assumptions failed contact with data)

1. **Structure is STAR fan-out, not a path.** `chained_links = 0` in all 19 campaigns — no hop where a target becomes the next source. Dominant shape: one foothold (C17693) → many high-in-degree targets. Top-2 campaigns (261, 207 events = 72% of fit) are 1→168 and 1→88 fan-outs, 13.6h / 9.7h.
2. **`chain_length` = fan-out breadth, not path depth.** Sampling "261" as a 261-hop walk would dead-end constantly. Median fit chain-length 15 (SPEC guessed ~9) is inflated by mega-fan-outs.
3. **Attributes are constant.** No within-positive variance to model. Day-3 conditional table collapses to emitting `(NTLM, Network, LogOn, Success)`.

## Decisions (operator confirmed at gate)

- **Day-2 structure generator = branching fan-out (option a)**, NOT SPEC's linear random walk.
  Seed at a foothold (high out-degree), sample **fan-out breadth** from the real per-campaign dst-count distribution (168, 88, 46, 17, 12, …), emit that many outward auths to real out-edge targets weighted by in-degree, harvesting credentials. Keep short 1–2 hop excursions only if mixed campaigns warrant. Hard edge constraint unchanged (every edge real).
- **Attributes = constant** `(NTLM, Network, LogOn, Success)`. No conditional sampling machinery needed (Day-3 simplifies).
- **Scope = full (option i):** fan-out generator → 100k+ events → TSTR with all 3 baselines. **Report the rarity-baseline comparison as the headline honest result** — synthetic uplift over the simple rarity heuristic may be modest; SPEC §5 accepts this.

## README framing corrections (apply on Day 4 writeup)

- **"Multi-hop lateral movement chain" → "credential-reuse fan-out from a foothold."** Fan-out IS lateral movement, but it is NOT a multi-hop path. Do not claim traversal/pivoting the data does not show.
- **Do not dress inference as fact.** Defensible: "a single source authenticates to many high-in-degree targets with many distinct credentials, uniformly via successful NTLM network logons, no observed multi-hop pivoting." Label mechanism (Mimikatz/pass-the-hash/escalation) as *interpretation* — LANL is anonymized auth events; it shows structure and outcome, not mechanism or intent. This is a sanctioned red-team exercise; C17693 is likely the operator console, creds likely provided not stolen.
- **State the transfer ceiling honestly.** TSTR works because fit and holdout are the *same* red team (same toolkit, same NTLM fingerprint, same fan-out shape). Generalization to a *different* attacker (Kerberos/WMI/low-and-slow) is NOT demonstrated — put it in Future Work / limitations.
- **Attribute constancy is tool-specific.** 100% NTLM/Network is this engagement's fingerprint, not a universal malicious signature. The *structural* signal (rare edge + high-in-degree target + credential novelty) generalizes; the attribute signal does not.

## Day-2 plan deltas (rewrite before executing Day 2)

- `walker.py` → fan-out generator. Replace "sample chain length, walk a path" with "sample fan-out breadth, emit outward auths from a foothold."
- Re-derive the profile field consumed by Day 2: use **per-campaign dst-count (breadth)** distribution, not events-per-campaign-as-depth.
- Dead-end guard rail still applies per emitted edge, but breadth-based generation dead-ends far less than deep paths.
- Day-3 attribute layer shrinks to a constant emitter + Δt bootstrap.

## Day-2 EXECUTED (2026-08-09) — result + open gaps for the Day-3 gate

**Status:** Day-2 fan-out generator built (`src/walker.py`), merged to `main`. All exit criteria PASS on real data: 10k campaigns / 33.7s, **0** edges out of graph, cap_rate 0.187 (<0.2), synth breadth median=9 / max=168 both exactly matching the real fit split (both mega-fan-outs 168/88 reproduced). 19/19 tests pass.

**What Day-2 is:** empirical-resample (breadth + Δt drawn directly from the 13 fit campaigns) + hard "every edge is real" graph constraint + hand-set weighting `edge_w^alpha × (indeg+1)^beta × credential_bonus`. **No training** — alpha=beta=1.0, credential_bonus=2.0 are default priors, not fitted.

**Four open gaps surfaced by grilling — carry into Day-3/Day-4, none block Day-2:**

1. **Credential semantics are a proxy, not replay.** A "credential" = the anonymized `src_user` account (`U####@DOM`), not a password/hash. Reuse is *real and measured* (campaign 5: 261 events / 168 targets / **45 distinct accounts** — few creds sprayed across many hosts, directly observed in the log). BUT the generator does **not** replay real account→target pairings: `harvest_credentials` reads `user_host` (built from **benign** `auth.txt`) as a proxy for "accounts harvestable on the foothold," and assigns per target via `host_users[t] ∩ compromised`. So reuse *statistics* are real; specific cred→target *assignments* are invented from benign co-occurrence. Assumption to state in the writeup: *account presence in normal logs ≈ what an attacker could steal.*

2. **`creds_per_campaign` measured but NOT enforced in generation.** Distinct-credential count per campaign `[45,39,22,13,12,4,4,2,2,2,1,1,1]` is computed and thrown away (labelled "Day-4 fidelity check"). Emitted distinct-user count is emergent. → Fix: after sampling breadth `k`, also sample cred-count `m` from this distribution, restrict harvest pool to `m`, so reuse *intensity* (few creds↔many targets vs many creds) matches real.

3. **Foothold pool too wide vs. observed seeds.** Real footholds = **4 hosts**, one dominant: C17693 (out-deg 534, 9 of 13 campaigns incl. both mega), plus C19932/C22409 (~30). Generator samples ∝ out-degree from **all 17,666** nodes with out-deg≥25 — preserves "high-out-degree dominates" but *invents footholds that never attacked.* `seed_hosts` is already in the profile. Gate decision: keep broad (generalize) vs. pin to observed seeds (mimic this operator).

4. **No wall-clock / hour-of-day placement — Day-3 job.** Events are `(dt_offset, user, src, dst)`, `dt_offset` = seconds from campaign start (0-based). *Relative* spacing IS captured (Δt bootstrap → reproduces the real low-and-slow spread: 261 events over 13.6h, 0.3 ev/min, NOT a post-login burst). *Absolute* placement is NOT: no date/hour/weekday. Real attacks fired in **business hours (LANL hr 6–17)** — deliberate blend-in. Day-1 captured the `hourly` aggregate but Day-2 doesn't consume it. → Day-3: map `dt_offset` → wall-clock by sampling campaign start from the real `hourly` distribution.

5. **Single-foothold-per-campaign assumption vs. multi-origin reality.** A real campaign is a **time-sessionized** cluster, NOT a foothold group — and **6 of 13 fit campaigns span 2–3 source computers** (camp 7 = 207 events / 88 targets across **3** `src_computer`; camp 8 = 3; camps 1/2/3/10 = 2). Only 7 of 13 are truly single-origin. The generator emits **one foothold per campaign**, so it cannot produce a multi-origin campaign — it renders camp-7-like activity as one machine → 88 targets instead of 3 machines → 88. Dominant foothold (C17693) carries most events so it's a fair approximation, but the model understates origin diversity. → Fix (Day-3/4 if fidelity matters): allow a campaign to draw a small foothold *set* (sample origin-count from the observed `src_computer`-per-campaign distribution) and partition breadth across them. Also clarifies schema: event = account `src_user` (`U#`) **from** `src_computer` (`C#`) **to** `dst_computer` (`C#`); graph edge is machine→machine, credential is the separate `src_user` axis (`dst_user == src_user` 100%).
