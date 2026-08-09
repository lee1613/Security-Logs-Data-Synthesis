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
