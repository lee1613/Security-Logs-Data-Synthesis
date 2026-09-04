# Data Insights — what the LANL auth corpus actually shows

Measured, not recalled. Every number here comes from `data/derived/` via a direct pass over
`graph.pkl`, `redteam_fit.csv`, `redteam_holdout.csv` and `benign_holdout.csv`.

Companion to [`validation_report.md`](validation_report.md), which covers the Day-4 experiment.
This file covers the **dataset**: what a row means, and what the distributions say.

---

## 0. What one row is

The corpus is LANL's authentication log — every credential validation at Los Alamos National Lab
across 58 days. Nine columns:

| column | meaning | example |
|---|---|---|
| `time` | **integer seconds since corpus start**, 0 – 5,011,200. Not a calendar date. | `1847858` |
| `src_user` | account presenting the credential | `U66@DOM1` |
| `dst_user` | account it resolves to (differs only under delegation) | `U66@DOM1` |
| `src_computer` | machine that initiated the request | `C17693` |
| `dst_computer` | machine that validated it / hosts the service | `C2725` |
| `auth_type` | protocol | `NTLM`, `Kerberos`, `?` |
| `logon_type` | session kind | `Network`, `Interactive`, `Service` |
| `auth_orientation` | half of the exchange | `LogOn`, `LogOff`, `TGS` |
| `success` | outcome | `Success` / `Fail` |

A row is **not** "a person sat down at a keyboard." It is *a credential was validated — who
presented it, and where the check happened.*

`DOM1` is the **domain** — the organization, not a machine. Both users and computers are members.
Machines have their own accounts, written with a trailing `$` (`C14917$@DOM1`), and authenticate on
their own behalf to fetch policy, register DNS, renew tickets — with no human present.

**The fact that makes this project's subject matter possible:** `U66@DOM1` is valid on all 17,666
machines, not just the one the person usually sits at. Credentials are domain-wide. Lateral movement
is exactly the exploitation of that: land on one machine, steal a credential from memory, reuse it
elsewhere. It works *only* because the credential is not tied to hardware.

### Scale

| | count |
|---|---|
| login events in corpus | **1,051,430,459** |
| distinct machines | 17,666 |
| machine-pairs that ever communicate | 419,744 |
| **red-team events** | **701** |
| red-team campaigns | 19 (13 fit / 6 holdout) |

**701 in 1.05 billion — 1 in 1.5 million.** This single ratio drives every design decision in the
project. You cannot train a conventional classifier on 701 examples, which is the entire premise for
attempting synthesis.

---

## 1. The attack attributes are *perfectly* constant — and that is a trap

| value | red team | benign |
|---|---|---|
| `auth_type = NTLM` | **100.00%** | 4.84% |
| `logon_type = Network` | **100.00%** | 83.32% |
| `auth_orientation = LogOn` | **100.00%** | 42.46% |
| `success = Success` | 99.86% | 99.38% |

All 701 attack events are `NTLM / Network / LogOn / Success`. Zero variance.

This *looks* like an excellent detection feature. It is a trap, and it is what wrecked the augmented
model in Day 4. NTLM covers 4.84% of benign traffic — **~51 million events** at corpus scale. It is
not "attacker behaviour"; it is this red team's tooling configured one way. A team using Kerberos
would be invisible to any detector riding this.

Benign `auth_type` for contrast: `?` 57.55%, `Kerberos` 35.70%, `NTLM` 4.84%, `Negotiate` 1.82%.

---

## 2. Over half of "benign" is not the same kind of event at all

| | red team | benign |
|---|---|---|
| self-auth (`src_computer == dst_computer`) | **0.00%** | **53.48%** |
| machine account (`C1234$@DOM1`) | **0.00%** | **60.32%** |
| `src_user == dst_user` | 100.00% | 98.72% |

More than half of the "normal" class is a machine authenticating **to itself** — screen unlocks,
service starts, scheduled tasks, cached-credential logons. The check never crossed the network.
Another large share is machine accounts doing automated housekeeping.

Lateral movement is by definition machine → *different* machine under a human account. So a large
fraction of the haystack is **not a candidate for the needle**.

**Caveat, stated because it matters:** this partly reflects how labels were made. `redteam.txt` lists
only cross-machine human-account logins, so "self-auth is benign" is true *by construction of the
label file*, not a discovered property. A detector leaning on it is partly detecting the labelling
convention.

---

## 3. Campaigns are wildly uneven, and there are only 19

| campaign | events | targets | sources | creds | duration |
|---:|---:|---:|---:|---:|---:|
| 5 | **261** | 168 | 1 | 45 | 13.6 h |
| 7 | 207 | 88 | 3 | 39 | 9.7 h |
| 8 | 75 | 46 | 3 | 22 | 9.0 h |
| 15 | 26 | 24 | 1 | 19 | 3.1 h |
| 9 | 25 | 17 | 1 | 13 | 6.1 h |
| … | … | … | … | … | … |
| 4, 12, 14 | **1** | 1 | 1 | 1 | 0 s |

**Medians: 9 events, 5 targets, 1 source machine, 3 credentials, ~3 hours.**

Two campaigns hold **67% of all attack events**. Three campaigns are a single event. The generator
fits its sampling distributions on 13 of these. *Fitting a distribution to 13 wildly-skewed samples
is fragile*, and that fragility sits upstream of everything downstream.

`distinct_sources` median = **1** confirms the campaign shape: one foothold fanning out to many
targets, **not** a chain of hops. (`chained_links = 0` in all 19 campaigns — this is what forced the
SPEC v1.2 fan-out correction.)

---

## 4. Graph position — the central finding

| | p25 | **median** | p75 |
|---|---:|---:|---:|
| **every machine** (in-degree) | 1 | **4** | 6 |
| machine a **benign event** targets | 955 | **14,199** | 14,323 |
| machine the **red team** targeted | 10 | **20** | 104 |
| machine the **synthetic generator** targeted | 143 | **11,659** | 14,221 |

Read carefully — these three facts are easy to conflate:

- The *typical machine* is obscure: median in-degree **4**.
- The *typical login event* targets a giant hub: median in-degree **14,199**. A handful of domain
  controllers and file servers absorb most of the billion events.
- The *red team* targeted machines at in-degree **20** — ordinary workstations, away from the hubs.

The network looks like this:

```
        few DOMAIN CONTROLLERS / SERVERS
        (in-degree ~14,000 — everyone talks to them)
                 ▲   ▲   ▲   ▲
                 │   │   │   │
       ┌─────────┴───┴───┴───┴─────────┐
       │     ~17,000 WORKSTATIONS      │
       │  (in-degree ~4 — talk to      │
       │   almost nobody but servers)  │
       └───────────────────────────────┘
```

Normal traffic is overwhelmingly **vertical** — workstations up to servers. Workstations rarely talk
**sideways** to each other. The red team moved `C17693 → C2725`: workstation to workstation, cutting
*across* the grain. Sideways traffic between two machines with no shared history is the anomaly, and
that is precisely what `edge_rarity` measures.

### The synthetic generator's failure, stated exactly

**It aimed attacks at the servers** (in-degree 11,659) — straight up the vertical arrows where all
legitimate traffic already flows. It manufactured *normal-shaped* traffic and labelled it "attack."

Not merely inaccurate: actively teaching the detector that hub-directed traffic — i.e. most
legitimate traffic — is malicious. That is why the augmented model abandoned graph position and fled
to `auth_type_NTLM`.

### Permission vs. habit

The deeper framing. **Permission is broad; behaviour is narrow.** A domain account *could* reach
hundreds of machines. In practice a person touches three or four, stably, for weeks. Multiply across
17,666 machines and a small set of pairs lights up constantly while the vast majority of possible
pairs never occur.

`edge_rarity` measures that gap between *permitted* and *observed*. Note that **every red-team login
was permitted** — real stolen credentials, domain approved every one, all returned `Success`.
Authentication worked exactly as designed. The only separator is prior history: one pair had 219,000
previous events, the other ~3.

**Rarity is not proof.** A new hire, a new project, or someone covering a colleague produces a
first-time pair that looks identical to lateral movement. Those are the ~8,900 benign holdout rows
ranking above the red team's 5th percentile — and why precision stays poor even when ranking is good.

This is the target any better generator must hit: reproducing *who may log in where* is easy and
useless, because the domain permits nearly everything. The hard part is reproducing **narrowness** —
stable habitual pairs, and an attacker's deviation that is small in count but structurally distinct.
The current generator samples a foothold's real out-edges weighted toward popular machines, which
reproduces **permission and popularity, not habit**.

---

## 5. Attacks start from moderately-connected machines

| | p25 | median | p75 |
|---|---:|---:|---:|
| red-team **footholds** (out-degree) | 30 | 30 | 157 |
| all machines (in-degree, for scale) | 1 | 4 | 6 |

Footholds have real reach — an admin workstation or jump box — then fan out to quiet places. The
generator's `min_foothold_out_degree: 25` (`config.yaml:36`) is consistent with this.

Median edge weights, the same contrast in a different unit:

| | median edge weight (events / 58 days) |
|---|---:|
| connection the red team used | **~3.6** |
| connection a benign event used | **~219,000** |

Five orders of magnitude.

---

## 6. Two strong signals exist in the data and are not features

| candidate | red team | benign | in `features.py`? |
|---|---|---|---|
| `is_self_auth` | 0.00% | 53.48% | **no** |
| `is_machine_account` | 0.00% | 60.32% | **no** |

Both are cheap and would remove a large share of the haystack immediately. Both carry the §2 caveat
about label construction. Neither is currently computed — `features.py` emits 4 structural columns
and 13 attribute one-hots, and these are not among them.

---

## What this implies

1. **Attribute columns cannot separate the classes.** A legitimate file-share access and a lateral
   movement are structurally identical rows: same protocol, same logon type, human account,
   machine → different machine, `Success`. Topology carries the distinction; attributes do not.
2. **The generator's target-selection is the primary defect**, and it is upstream of the `beta`
   knob — `beta=0` yields *uniform over permitted*, which still is not *habit*.
3. **Distribution-fitting on 13 skewed campaigns is a weak foundation** regardless of what is fitted.
4. **The haystack is padded with non-candidates**, so the nominal 1:4,033 base rate overstates
   difficulty in one direction while the sampling (§5 of the validation report) understates it far
   more in the other.

---

*Reproduce: figures here come from `graph.pkl` plus the four CSVs in `data/derived/`; the Day-1
red-team profile is in `data/derived/redteam_profile.json`.*
