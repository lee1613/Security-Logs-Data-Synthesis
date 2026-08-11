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
                   "credential_novelty", "n_hosts_for_cred")
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

    def arr(it):
        return np.fromiter(it, dtype=float, count=n)

    weight = arr(graph[s][d]["weight"] if graph.has_edge(s, d) else _UNSEEN_EDGE_WEIGHT
                 for s, d in zip(src, dst))
    return pd.DataFrame({
        "edge_rarity": np.log(total_w / weight),
        "dst_in_degree": arr(graph.in_degree(d) if d in graph else 0 for d in dst),
        "src_out_degree": arr(graph.out_degree(s) if s in graph else 0 for s in src),
        # polarity: 1 == NOVEL, i.e. this credential was NOT seen on this host in
        # benign history. Rarity-flavoured, same direction as edge_rarity.
        "credential_novelty": arr(0.0 if u in host_users.get(d, ()) else 1.0
                                  for u, d in zip(usr, dst)),
        "n_hosts_for_cred": arr(cred_hosts.get(u, 0) for u in usr),
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
