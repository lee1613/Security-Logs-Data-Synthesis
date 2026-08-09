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
