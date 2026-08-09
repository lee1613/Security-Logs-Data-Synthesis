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


def test_build_mixed_corpus_downsamples_malicious_when_abundant():
    # real Day-3 regime: malicious events >> what the target rate allows into benign
    mal = _mal_rows(1000)
    benign = _benign_rows(100000)
    mixed, labels = M.build_mixed_corpus(mal, benign, rate=0.001, seed=3)
    n_mal = int((labels["is_malicious"] == 1).sum())
    n_ben = int((labels["is_malicious"] == 0).sum())
    assert n_ben == 100000                       # benign held at full pool
    assert abs(n_mal - 100) <= 1                  # mal down-sampled to rate*benign = 100
    assert len(mixed) == len(labels)
    assert list(mixed["time"]) == sorted(mixed["time"])


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
