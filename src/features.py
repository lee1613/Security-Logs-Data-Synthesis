"""Day-4 feature builder: one auth row -> one feature vector.

The detector built on these features is a MEASURING INSTRUMENT for synthetic-data
quality, not a detection-science contribution. The `groups` argument is the
ablation switch: dropping "attribute" re-runs the whole comparison without the
NTLM/Network one-hots, which tells us whether a detected signal is real graph
STRUCTURE or just this red team's tooling artifact.

LEAKAGE RULE (the whole point of this module): every feature comes from the
benign graph + fit split ONLY. `build_features` is a pure ROW-LOCAL MAP -- each
output row is a function of that single input row plus `graph` and `host_users`,
and nothing else. It computes no statistic over `rows_df` (no mean, no
value_counts, no fitted encoder, no vocabulary inferred from the input), so
passing holdout rows in the same frame cannot move any other row's values.
Nothing here can reach a holdout file.

WHY `credential_novelty` WAS SPECIFIED AND THEN REMOVED (Day-4 finding, keep this)
----------------------------------------------------------------------------------
The Day-4 plan listed a fifth structural feature, `credential_novelty` -- binary,
1 if this `src_user` had NOT been seen on this `dst_computer` in benign history.
It was built, measured, and dropped. Measured values:

    redteam_fit (all 650 rows)      credential_novelty = 0.0000
    benign_fit  (first 200k rows)   credential_novelty = 0.0000
    synth_auth  (first 20k rows)    credential_novelty = 0.3664

It is DEFINITIONALLY zero on any real row. `parse.compute_user_host_counts`
builds `aggregates["user_host"]` by unioning (src_user, src_computer) with
(src_user, dst_computer) over the FULL auth corpus -- and the real red-team
events are inside that corpus. So for any row drawn from real data the pair is
tautologically already observed. It is nonzero only on synthetic rows, where 1
marks the fallback branch in `walker.generate_campaign` (`pool =
list(compromised)`, taken when no harvested credential was ever seen on the
target). That makes it a "this row is synthetic" marker, not an attack signal:
left in, a detector trained on real+synth in the augmented arm of the scarcity
curve could learn a rule keyed on a feature that is dead at evaluation time,
corrupting the headline experiment.

The graph and the user_host map are deliberately NOT decontaminated. In reality
the log does contain the undetected attack, so the contaminated history is the
realistic one; scrubbing it would be optimistically unrealistic. `edge_rarity`
is kept for the same reason -- and it separates hard: mean 19.238 on redteam_fit
vs 8.198 on benign_fit.

`n_hosts_for_cred` is kept: its synth-vs-real gap (mean 2,319 on synth vs 261 on
redteam_fit and 231 on benign_fit) is a genuine generator-fidelity gap with
overlapping distributions, not a definitional artifact. It stays a legitimate
feature and is reported as a finding.
"""
import numpy as np
import pandas as pd

# One-hot vocabularies. FIXED-WIDTH and module-level on purpose: the column set
# must not depend on which rows are passed, or the train and eval matrices will
# not align. Derived once from data/derived/benign_fit.csv (FIT SIDE ONLY, never
# holdout, never the input frame) by frequency; anything below 0.1% of benign_fit
# is folded into `_other` rather than emitting a near-dead column. The dict maps
# raw value -> column suffix.
_AUTH_TYPE_VOCAB = {"?": "unknown", "Kerberos": "Kerberos",   # .576 / .355
                    "NTLM": "NTLM", "Negotiate": "Negotiate"}  # .050 / .018
_LOGON_TYPE_VOCAB = {"Network": "Network", "?": "unknown", "Service": "Service",
                     "Unlock": "Unlock", "Interactive": "Interactive",
                     "Batch": "Batch", "NewCredentials": "NewCredentials"}

# An edge absent from the benign graph is scored as if it had been seen half a
# time: rarity = log(total_weight / 0.5) = log(2 * total_weight), which is
# strictly greater than the rarest observed edge (log(total_weight / 1)) and
# always finite. No NaN, no silent 0.0.
_UNSEEN_EDGE_WEIGHT = 0.5

STRUCTURAL_COLS = ("edge_rarity", "dst_in_degree", "src_out_degree",
                   "n_hosts_for_cred")
ATTRIBUTE_COLS = tuple(
    [f"auth_type_{s}" for s in list(_AUTH_TYPE_VOCAB.values()) + ["other"]]
    + [f"logon_type_{s}" for s in list(_LOGON_TYPE_VOCAB.values()) + ["other"]])
_GROUPS = ("structural", "attribute")   # canonical emit order

# Deliberately EXCLUDED: hour_of_day (benign peaks in the same business hours and
# there are zero off-hours attack examples -- it survives as a distributional
# figure, never as a feature), success and auth_orientation (constant on both
# sides => zero information). Do not add "helpful" extras: every additional
# feature weakens the ablation's interpretability.


def _user_host_counts(host_users):
    """host -> set(users) inverted to user -> number of distinct hosts.
    ponytail: rebuilt per call (~0.3s on the real 17.7k-host map). Memoize only
    if build_features ends up in a hot loop; a cache keyed off a plain dict is
    the kind of thing that quietly goes stale."""
    counts = {}
    for users in host_users.values():
        for u in users:
            counts[u] = counts.get(u, 0) + 1
    return counts


def _structural(rows_df, graph, host_users):
    n = len(rows_df)
    src = rows_df["src_computer"].to_numpy()
    dst = rows_df["dst_computer"].to_numpy()
    usr = rows_df["src_user"].to_numpy()
    total_w = sum(d["weight"] for _, _, d in graph.edges(data=True)) or 1.0
    cred_hosts = _user_host_counts(host_users)

    def lookup(keys, table):
        """Vectorized dict lookup; a key absent from the table degrades to 0.
        ~7x faster than a per-row generator on 500k rows, which matters because
        the scarcity sweep calls build_features 50+ times."""
        return pd.Series(keys).map(table).fillna(0).to_numpy(dtype=float)

    # edge weight keys on the (src,dst) PAIR, so it stays a per-row lookup
    weight = np.fromiter(
        (graph[s][d]["weight"] if graph.has_edge(s, d) else _UNSEEN_EDGE_WEIGHT
         for s, d in zip(src, dst)), dtype=float, count=n)
    return pd.DataFrame({
        "edge_rarity": np.log(total_w / weight),
        "dst_in_degree": lookup(dst, dict(graph.in_degree())),
        "src_out_degree": lookup(src, dict(graph.out_degree())),
        # NOTE: no credential_novelty here -- see the module docstring for why the
        # plan specified it and why it was measured and removed.
        "n_hosts_for_cred": lookup(usr, cred_hosts),
    }, index=rows_df.index)


def _one_hot(values, vocab, prefix):
    suffix = pd.Series(values).map(vocab).fillna("other")
    return {f"{prefix}_{s}": (suffix == s).to_numpy(dtype=float)
            for s in list(vocab.values()) + ["other"]}


def _attribute(rows_df):
    cols = _one_hot(rows_df["auth_type"], _AUTH_TYPE_VOCAB, "auth_type")
    cols.update(_one_hot(rows_df["logon_type"], _LOGON_TYPE_VOCAB, "logon_type"))
    return pd.DataFrame(cols, index=rows_df.index)


def build_features(rows_df, graph, host_users, *, groups=("structural", "attribute")):
    """Feature matrix for any auth rows in AUTH_COLS order (synth, real red team,
    or benign). One output row per input row, index-aligned, numeric, deterministic.
    `groups` selects which families to emit -- this is the ablation switch."""
    if not groups:
        # an empty sequence has no unknown members, so it slips past the check
        # below and dies inside pd.concat with a message that names nothing
        raise ValueError(f"groups must not be empty; expected some of {_GROUPS}")
    unknown = [g for g in groups if g not in _GROUPS]
    if unknown:
        raise ValueError(f"unknown feature group(s) {unknown}; expected {_GROUPS}")

    # emit in canonical order so column layout never depends on caller ordering
    parts = []
    if "structural" in groups:
        parts.append(_structural(rows_df, graph, host_users))
    if "attribute" in groups:
        parts.append(_attribute(rows_df))
    return pd.concat(parts, axis=1)
