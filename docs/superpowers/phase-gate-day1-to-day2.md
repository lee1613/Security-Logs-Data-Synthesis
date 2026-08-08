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
