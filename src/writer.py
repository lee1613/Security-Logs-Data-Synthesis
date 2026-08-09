"""Day-3 writer: constant attributes, time placement, LANL-format emission,
labels. Attributes are constant by measurement (SPEC v1.2): every synthetic
malicious auth is (NTLM, Network, LogOn, Success) with dst_user == src_user.
"""
import csv

import numpy as np
import pandas as pd

# Constant attribute tuple (SPEC v1.2 — fit split is 100% these values).
AUTH_TYPE = "NTLM"
LOGON_TYPE = "Network"
AUTH_ORIENTATION = "LogOn"
SUCCESS = "Success"

# Exact LANL field order (matches src/parse.AUTH_COLS).
AUTH_FIELDS = ["time", "src_user", "dst_user", "src_computer", "dst_computer",
               "auth_type", "logon_type", "auth_orientation", "success"]
LABEL_FIELDS = ["row_id", "is_malicious", "campaign_id", "hop_index"]

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


def sample_start_time(hourly_df, rng, max_offset, collection_seconds):
    """Absolute campaign start (seconds), hour-of-day ∝ real hourly volume,
    day uniform, kept so start + max_offset stays inside the collection window."""
    hours = hourly_df["hour"].to_numpy()
    p = hourly_df["cnt"].to_numpy(dtype=float)
    p = p / p.sum()
    latest_start = max(collection_seconds - max_offset, SECONDS_PER_DAY)
    n_days = max(latest_start // SECONDS_PER_DAY, 1)
    while True:
        day = int(rng.integers(0, n_days))
        hour = int(rng.choice(hours, p=p))
        sec = int(rng.integers(0, SECONDS_PER_HOUR))
        start = day * SECONDS_PER_DAY + hour * SECONDS_PER_HOUR + sec
        if start + max_offset <= collection_seconds:
            return start


def place_campaign(events, start):
    """Shift relative (dt_offset, user, src, dst) events to absolute time."""
    return [(int(start + off), user, src, dst) for off, user, src, dst in events]
