"""Day-3 mixed corpus: inject synthetic malicious rows into real benign at a
target base rate, sorted by time. Labels row-aligned (1 malicious, 0 benign)."""
import numpy as np
import pandas as pd

import src.writer as Wr

_LABEL_COLS = ["row_id", "is_malicious", "campaign_id", "hop_index"]


def build_mixed_corpus(mal_df, benign_df, rate, seed):
    """rate = malicious:benign. Shrink whichever class is over target (never
    upsample): if rate*len(benign) <= len(mal), hold benign full and down-sample
    malicious to rate*n_benign (the real Day-3 regime — abundant malicious);
    otherwise hold malicious full and down-sample benign to n_mal/rate. Concat,
    sort by time. Returns (mixed_auth_df, labels_df), row-aligned."""
    rng = np.random.default_rng(seed)
    n_m, n_b = len(mal_df), len(benign_df)
    if rate * n_b <= n_m:
        n_benign, n_mal = n_b, int(round(rate * n_b))
    else:
        n_mal, n_benign = n_m, min(n_b, int(round(n_m / rate)))

    mi = rng.choice(n_m, size=n_mal, replace=False)
    bi = rng.choice(n_b, size=n_benign, replace=False)
    mal = mal_df.iloc[mi][Wr.AUTH_FIELDS].copy()
    mal["is_malicious"] = 1
    benign = benign_df.iloc[bi][Wr.AUTH_FIELDS].copy()
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
