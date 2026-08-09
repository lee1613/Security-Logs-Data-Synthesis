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
